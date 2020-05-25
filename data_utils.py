# -*- coding: utf-8 -*-

import os
import csv
import fnmatch

import numpy as np
from dataset_generator import DataGenerator






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


# TODO: This may be covered by np.swapaxes
def exchange_channels(data):
    return data.reshape(
        (data.shape[0], data.shape[2] * data.shape[1])).reshape((
            data.shape[0], data.shape[2], data.shape[1]), order = 'F')



def unroll_generator(generator):
    first_batch = True
    for batch_x, batch_y in generator:
        if len(batch_y) == 0:
            continue

        if first_batch:
            first_batch = False
            x_set = batch_x
            y_set = batch_y
        else:
            x_set = np.concatenate((x_set, batch_x))
            y_set = np.concatenate((y_set, batch_y))
    return x_set, y_set


def prepare_generators(
        path_data,
        main_label_dict,
        label_dict,
        split_ratio,
        ignore_labels = []):

    dirs = os.listdir(path_data)
    files_csv = fnmatch.filter(dirs, "*.csv")

    main_labels = list_main_labels(path_data, files_csv, main_label_dict)

    binarizer_dict  = binarize_labels(label_dict)

    id_split = split_dataset(split_ratio, main_labels)
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
            data_path = path_data,
            to_fit=True,
            n_average = 50,
            batch_size = 10,
            iter_per_epoch = 10,
            up_sampling = True,
            n_timepoints = 501,
            n_channels=30,
            # n_classes=1,
            shuffle=True)
        generators.append(generator)
    return generators


def binarize_labels(label_dict):
    values = [int(x) for x in label_dict.values()]
    size = np.max(values)
    binarizer_dict = dict()
    for value in set(label_dict.values()):
        bin_value = [0] * (size+1)
        bin_value[int(value)] = 1
        binarizer_dict[value] = bin_value
    return binarizer_dict


def list_main_labels(path_data, files_csv, main_label_dict):
    label_collection = []

    for filename in files_csv:
        y_EEG = read_csv_labels(filename, path_data)
        labels_unique = list(set(y_EEG))
        label_collection.append(labels_unique)


    main_labels = [main_label_dict.get(x[0]) for x in label_collection]
    return main_labels


def read_csv_labels(filename, PATH):
    metadata = []
    filename = os.path.join(PATH, filename)
    with open(filename, 'r') as readFile:
        reader = csv.reader(readFile, delimiter=',')
        for row in reader:
            metadata += row
            # if len(row) > 0:
                # metadata.append(''.join(row)) # TODO Delete, hotfix for csv issue
    readFile.close()

    return metadata


def split_dataset(split_ratio, main_labels):
    # id_split = [[]]*3
    ids_train = []
    ids_val = []
    ids_test = []

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

        ids_train.extend(list(trainIDs))
        ids_val.extend(list(valIDs))
        ids_test.extend(list(testIDs))

    np.random.shuffle(ids_train)
    np.random.shuffle(ids_val)
    np.random.shuffle(ids_test)
    return ids_train, ids_val, ids_test