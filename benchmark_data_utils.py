#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:18:48 2020

@author: jack
"""

import numpy as np
import os
from sklearn import preprocessing

def load_data(path_data, ts_type, n_samples, ignore_noise):
    noise = not ignore_noise
    x_data = np.load(os.path.join(
        path_data, f"x_data_{ts_type}_s{n_samples}_n{int(noise)}.npy"))
    y_data = np.load(os.path.join(
        path_data, f"y_data_{ts_type}_s{n_samples}_n{int(noise)}.npy"))
    data_type = f"{ts_type}-noise{int(noise)}"
    return x_data, y_data, data_type

def collect_labels(y_data):
    # Separate data by labels

    label_collection = np.unique(y_data)

    label_ids_dict = dict()
    for label in label_collection:
        label_ids_dict[label] = list()

    for i in range(len(y_data)):
        label = y_data[i]
        label_ids_dict[label] = label_ids_dict[label] + [i]
    return label_collection, label_ids_dict


def split_labels(label_collection, label_ids_dict, seed = None):

    if seed is not None:
        np.random.seed(seed)
    split_ratio = (0.7, 0.15, 0.15)

    ids_train = []
    ids_val = []
    ids_test = []

    for label in label_collection:
        indices = label_ids_dict[label]
        n_label = len(indices)
        print("Found", n_label, "datapoints for label", label)

        n_train = int(split_ratio[0] * n_label)
        n_val = int(split_ratio[1] * n_label)
        n_test = n_label - n_train - n_val
        print("Split dataset for label", label, "into train/val/test fractions:",
              n_train, n_val, n_test)

        # Select training, validation, and test IDs:
        trainIDs = np.random.choice(indices, n_train, replace=False)
        valIDs = np.random.choice(
            list(set(indices) - set(trainIDs)), n_val, replace=False)
        testIDs = list(set(indices) - set(trainIDs) - set(valIDs))

        ids_train.extend(list(trainIDs))
        ids_val.extend(list(valIDs))
        ids_test.extend(list(testIDs))
    return ids_train, ids_val, ids_test


def binarize_labels(y_data, label_collection):
    lb = preprocessing.LabelBinarizer()
    lb.fit(label_collection)
    return lb.fit_transform(y_data)