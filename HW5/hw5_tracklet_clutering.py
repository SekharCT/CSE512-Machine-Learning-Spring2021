
"""
This is a dummy file for HW5 of CSE353 Machine Learning, Fall 2020
You need to provide implementation for this file

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 26-Oct-2020
Last modified: 26-Oct-2020
"""

import random
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import numpy as np
from hw5 import *


class TrackletClustering(object):
    """
    You need to implement the methods of this class.
    Do not change the signatures of the methods
    """

    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.model = None
        self.input_feature = []
        self.max = 40

    def __get_center(self,box):
        center_x = (box[0] + box[2])/2
        center_y = (box[1] + box[3])/2

        return center_x , center_y 


    def __feature_extract(self, tracklet):
        direction = []
        for v_det in tracklet["tracks"]:
            frm_id = v_det[0]
            bbox = v_det[1:]
            direction.extend(self.__get_center(bbox))

        if len(direction) < self.max:
            last_val = direction[-1]
            extend_list = [last_val for i in range(self.max-len(direction))]
            direction.extend(extend_list)
        
        elif len(direction) > self.max :
            direction = direction[:self.max]

        return np.array(direction)

    def add_tracklet(self, tracklet): 
        self.input_feature.append(self.__feature_extract(tracklet))

    def build_clustering_model(self):
        #self.model = TimeSeriesKMeans(n_clusters=self.num_cluster , metric="dtw", max_iter=10).fit(to_time_series_dataset(self.input_feature))
        self.model =  k_means(np.array(self.input_feature) ,k = self.num_cluster)   

    def get_cluster_id(self, tracklet):
        """
        Assign the cluster ID for a tracklet. This funciton must return a non-negative integer <= num_cluster
        It is possible to return value 0, but it is reserved for special category of abnormal behavior (for Question 2.3)
        """
        feature_vector = self.__feature_extract(tracklet)
        #cluster_id = self.model.predict(to_time_series_dataset(feature_vector)) + 1
        cluster_id = assignment(feature_vector.reshape(1,-1), self.model[0]) + 1
        # remove outliers based on score.

        return int(cluster_id[0])
