#!/usr/bin/env python
# coding: utf-8

"""
Test code written by Viresh Ranjan

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""

import copy
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import MincountLoss, PerturbationLoss
from utils import *
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import torch.optim as optim
import csv


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the FSC147 dataset")
parser.add_argument("-ts", "--test_split", type=str, default='val', choices=["val_PartA","val_PartB","test_PartA","test_PartB","test", "val"], help="what data split to evaluate on")
parser.add_argument("-m",  "--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth", help="path to trained model")
parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'annotation_Val_Test_384_VarV2.json'
data_split_file = data_path + 'Train_Test_Val_FSC147_HW6_Split.json'
im_dir = data_path + 'images_384_VarV2'
Q2_b_dir = data_path + 'Q2_b'
mask_dir = data_path + 'mask_images/mask_images'

with open('sample_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "Count"])

if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    print("Current setting is {}, but the image dir and annotation file do not exist.".format(args.data_path))
    print("Aborting the evaluation")
    exit(-1)

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    print("===> Using CPU mode.")
else:
    use_gpu = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
if use_gpu: resnet50_conv.cuda()
resnet50_conv.eval()

regressor = CountRegressor(6, pool='mean')
regressor.load_state_dict(torch.load(args.model_path))
if use_gpu: regressor.cuda()
regressor.eval()

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)


cnt = 0
SAE = 0  # sum of absolute errors
SSE = 0  # sum of square errors
output_dict = {}
result_output = []
boxes_dict = {}
imgs_dict = {}
gt_dict = {}

print("Evaluation on {} data".format(args.test_split))
im_ids = data_split[args.test_split]
pbar = tqdm(im_ids)
for im_id in pbar:
    anno = annotations[im_id]
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])

    rects = list()
    for bbox in bboxes:
        x1, y1 = bbox[0][0], bbox[0][1]
        x2, y2 = bbox[2][0], bbox[2][1]
        rects.append([y1, x1, y2, x2])

    image = Image.open('{}/{}'.format(im_dir, im_id))
    image.load()
    sample = {'image': image, 'lines_boxes': rects}
    sample = Transform(sample)
    image, boxes = sample['image'], sample['boxes']

    # mask 
    mask = Image.open('{}/{}_anno.png'.format(mask_dir, im_id.rsplit( ".", 1 )[ 0 ]))
    mask.load()
    #mask = torch.from_numpy(mask)
    sample = {'image': mask, 'lines_boxes': rects}
    sample = Transform(sample)
    mask, boxes_ = sample['image'], sample['boxes']
    mask = 1.0 - mask
    mask = mask[0,:,:]
    #print(mask.size())

    if use_gpu:
        image = image.cuda()
        boxes = boxes.cuda()
        mask = mask.cuda()

    with torch.no_grad(): features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

    if not args.adapt:
        with torch.no_grad(): output = regressor(features)
    else:
        features.required_grad = True
        adapted_regressor = copy.deepcopy(regressor)
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)
        for step in range(0, args.gradient_steps):
            optimizer.zero_grad()
            output = adapted_regressor(features)
            lCount = args.weight_mincount * MincountLoss(output, boxes, mask)
            lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
            Loss = lCount + lPerturbation
            # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
            # So Perform gradient descent only for non zero cases
            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()
        features.required_grad = False
        output = adapted_regressor(features)

    gt_cnt = dots.shape[0]
    pred_cnt = output.sum().item()
    cnt = cnt + 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err**2

    pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.\
                         format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))

    with open('sample_2.csv', 'a+', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([im_id, pred_cnt])
    # result_output.append([im_id, gt_cnt, pred_cnt])
    # output_dict[im_id] = output
    # boxes_dict[im_id] = boxes
    # imgs_dict[im_id] = image
    # gt_dict[im_id] = gt_cnt

    print("")

# result_output = np.array(result_output)
# count = []
# for i in range(len(result_output)):
#   count.append(np.float(result_output[:,2][i]) - np.float(result_output[:,1][i]))

# sorted_indices = np.argsort(count)
# result_output = np.array(result_output)
# over_cnt_imgs = result_output[:,0][sorted_indices[-5:]]
# under_cnt_imgs = result_output[:,0][sorted_indices[:5]]

# for i in range(5):
#   over_img = over_cnt_imgs[i]
#   under_img = under_cnt_imgs[i]
#   rslt_file_over = "{}/{}_over_out_gt_{}.png".format(Q2_b_dir, over_img, gt_dict[over_img])
#   rslt_file_under = "{}/{}_under_out_gt_{}.png".format(Q2_b_dir, under_img, gt_dict[under_img])
#   visualize_output_and_save(imgs_dict[over_img].detach().cpu(),output_dict[over_img].detach().cpu(),boxes_dict[over_img].cpu(),rslt_file_over)
#   visualize_output_and_save(imgs_dict[under_img].detach().cpu(),output_dict[under_img].detach().cpu(),boxes_dict[under_img].cpu(),rslt_file_under)

print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
