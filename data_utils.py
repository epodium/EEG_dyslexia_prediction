# -*- coding: utf-8 -*-

import os
import csv
import fnmatch

import numpy as np
from dataset_generator import DataGenerator



def collect_labels():
    label_collection = []
    label_counts = []

    for filename in files_csv:
        y_EEG = read_labels(filename, PATH_DATA_processed)
        labels_unique = list(set(y_EEG))
        label_collection.append(labels_unique)

        # Count instances for each unique label
        label_count = []
        for label in labels_unique:
            idx = np.where(np.array(y_EEG) == label)[0]
            label_count.append(len(idx))
        label_counts.append(label_count)


def read_csv_labels(filename, PATH):
    metadata = []
    filename = os.path.join(PATH, filename)
    with open(filename, 'r') as readFile:
        reader = csv.reader(readFile, delimiter=',')
        for row in reader:
            if len(row) > 0:
                # metadata.append(row)
                metadata.append(''.join(row)) # TODO Delete, hotfix for csv issue
    readFile.close()

    return metadata


def split_dataset(main_labels, split_ratio):
    IDs_train = []
    IDs_val = []
    IDs_test = []

    for label in list(set(main_labels)):
        idx = np.where(np.array(main_labels) == label)[0]
        N_label = len(idx)
        print("Found", N_label, "datapoints for label", label)

        N_train = int(split_ratio[0] * N_label)
        N_val = int(split_ratio[1] * N_label)
        N_test = N_label - N_train - N_val
        print("Split dataset for label", label, "into train/val/test fractions:", N_train, N_val, N_test)

        # Select training, validation, and test IDs:
        trainIDs = np.random.choice(idx, N_train, replace=False)
        valIDs = np.random.choice(list(set(idx) - set(trainIDs)), N_val, replace=False)
        testIDs = list(set(idx) - set(trainIDs) - set(valIDs))

        IDs_train.extend(list(trainIDs))
        IDs_val.extend(list(valIDs))
        IDs_test.extend(list(testIDs))

    return IDs_train, IDs_val, IDs_test


def prepare_set(indices, full_x_data, full_y_data):
    x_set = list()
    y_set = list()
    for idx in indices:
        x_set.append(full_x_data[idx])
        y_set.append(full_y_data[idx])
    x_set = np.array(x_set)
    # x_set = x_set.reshape(np.concatenate((x_set.shape, [1])))
    y_set = np.array(y_set)
    return x_set, y_set


def exchange_channels(data):
    return data.reshape(
        (data.shape[0], data.shape[2] * data.shape[1])).reshape((
            data.shape[0], data.shape[2], data.shape[1]), order = 'F')


def prepare_generators(path_data):

    main_labels # list of all main labels.
    label_dict #Dictionary for how to ouput the found labels to the model.
    binarizer_dict #Dictionary for how to ouput the found labels to the model.
    ignore_labels



    dirs = os.listdir(path_data)
    files_csv = fnmatch.filter(dirs, "*.csv")

    generators = []
    for id_list in id_split:
        generator = DataGenerator(
            list_IDs = id_list,
            main_labels = main_labels,
            label_dict = label_dict,
            binarizer_dict = binarizer_dict,
            ignore_labels = ignore_labels,
            path_data = path_data,
            filenames = [x[:-4] for x in files_csv],
            data_path = path_data, # TODO Repeated?
            to_fit=True,
            n_average = 40,
            batch_size = 32,
            iter_per_epoch = 50,
            up_sampling = True,
            n_timepoints = 501,
            n_channels=30,
            n_classes=1,
            shuffle=True)
        generators.append(generator)
    return generators
