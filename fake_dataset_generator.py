#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:04:10 2020

@author: breixo
"""

from tensorflow.keras.utils import Sequence
import numpy as np

class FakeDataGenerator(Sequence):
    
    def __init__(
            self,
            x_set,
            y_set,
            # functions_dict,
            batch_size=32):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        # self.functions_dict = functions_dict


    def __getitem__(self, index):
        batch_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        return batch_x, batch_y


    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    # def generate_batch(self, index):
    #     batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
    #     for i in range(len(batch_y)):
    #         item = self.modify_item(X[i], y[i])
    #         X[i] = item
    #     return x, y


    # def modify_item(self, o_item, bin_label):
    #     functions = self.functions_dict[str(bin_label)]
    #     return functions


