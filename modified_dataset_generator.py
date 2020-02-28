#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:04:10 2020

@author: breixo
"""

from dataset_generator import DataGenerator

class ModifiedDataGenerator(DataGenerator):
    
#    def __init__(
#            self, 
#            list_IDs,
#            main_labels,
#            label_dict,
#            binarizer_dict,
#            ignore_labels,
#            path_data,
#            filenames,
#            data_path, 
#            to_fit=True, #TODO: implement properly for False
#            n_average = 30,
#            batch_size=32,
#            iter_per_epoch = 2,
#            up_sampling = True,
#            n_timepoints = 501,
#            n_channels=30,
#            include_baseline = False,
#            subtract_baseline = False,
#            baseline_label = None,
#            shuffle=True,
#            warnings=False):
#        super().__init__(
#                list_IDs,
#                main_labels,
#                label_dict,
#                binarizer_dict,
#                ignore_labels,
#                path_data,
#                filenames,
#                data_path, 
#                to_fit,
#                n_average,
#                batch_size,
#                iter_per_epoch,
#                up_sampling,
#                n_timepoints,
#                n_channels,
#                include_baseline,
#                subtract_baseline,
#                baseline_label,
#                shuffle,
#                warnings)
        
        
    
    def __get_item__(self, index):
        if self.to_fit:
            original_X, y = super().__get_item__(index)
        else:
            original_X = super().__get_item__(index)
        X = self.modify_item(original_X)
        if self.to_fit:
            return X, y
        else:
            return X


    def modify_item(self, original_X):
        X = original_X # TODO Modify
        return X