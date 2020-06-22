#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:07 2020

@author: Breixo Solino
"""

# In[Import packages]

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, \
    UpSampling2D, Reshape, PReLU, Dropout, Lambda, Layer
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from real_data_utils import prepare_set, exchange_channels
import benchmark_data_utils


# In[Configure experiment]

from config import PATH_CODE, PATH_DATA, ROOT

model_name = "prototype_autoencoder"

output_file = os.path.join(PATH_CODE, 'models_trained', "autoencoder" , f"{model_name}.hdf5")

# real_data = True
real_data = False

if real_data:
    dataset_folder = 'processed_data_17mn'
    dataset_folder = 'processed_data_29mnd'
    # dataset_folder = "processed_data_41mnd"
else:

    # ## Load pre-processed dataset
    # + See notebook for preprocessing: ePODIUM_prepare_data_for_ML.ipynb.ipynb


    # ts_type = "test"
    ts_type = "benchmark1"
    # n_samples = 200
    n_samples = 1000
    ignore_noise = False
    # ignore_noise_network = True

    PATH_DATA_processed = os.path.join(ROOT, "test_data")


# In[Acquire data]
if real_data:
    pass
else:
    from tensorflow.keras.datasets import mnist
    (x_set_train, _), (x_set_val, _) = mnist.load_data()
    x_set_train = x_set_train.astype('float32') / 255.
    x_set_val = x_set_val.astype('float32') / 255.

    # x_data, y_data, data_type = benchmark_data_utils.load_data(
    #     PATH_DATA_processed,
    #     ts_type,
    #     n_samples,
    #     ignore_noise)

    # # Separate data by labels
    # label_collection, label_ids_dict = benchmark_data_utils.collect_labels(y_data)


    # # Split labels
    # ids_train, ids_val, ids_test = benchmark_data_utils.split_labels(label_collection, label_ids_dict, 1098)


    # # Randomize ids
    # np.random.shuffle(ids_train)
    # np.random.shuffle(ids_val)
    # np.random.shuffle(ids_test)


    # # Split datasets
    # # NOTE: mcfly takes data in the shape of length timeseries and channels

    # print(x_data.shape)
    # x_data = exchange_channels(x_data)
    # print(x_data.shape)

    # x_set_train = x_data[ids_train]
    # x_set_val = x_data[ids_val]
    # # x_set_test = x_data[ids_test]

    input_shape = x_set_train.shape


# In[Configure autoencoder]

IMPROVE = False
dim_length = input_shape[1]  # number of samples in a time series
dim_channels = input_shape[2]  # number of channels
weightinit = 'lecun_uniform'  # weight initialization
regularization_rate = 10 ** -2
learning_rate = 10 ** -4
metrics = ["accuracy"]


# In[Define autoencoder]

tf.keras.backend.clear_session()

inputs = Input(shape = (dim_length, dim_channels))
reshape = Reshape(target_shape=(dim_length, dim_channels, 1))(inputs)

previous_block = reshape
# filters = [(128, 5), (256, 11), (512, 21)]
# filters = [(16, 5), (32, 11), (64, 21)]
filters = [(32, (3, 3)), (64, (3, 3)), (128, (3, 3))]
autoencoder_filters = filters + list(reversed(filters))
shapes = [previous_block.shape]


class Interpolation(Layer):

    def __init__(self, target_shape):
        super(Interpolation, self).__init__()
        print(target_shape)
        # self.target_shape = tf.Variable(initial_value=target_shape,
        #                                 trainable=False)
        # global t_shape
        # t_shape = tf.Variable(initial_value=target_shape,
        #                                 trainable=False)
        self.target_shape = target_shape

    def call(self, inputs):
        # print(self.target_shape.read_value())
        # x = tf.image.resize(inputs, self.target_shape.read_value())
        # print(x.shape)
        x = tf.image.resize(inputs, self.target_shape)
        return x

    def get_config(self):
        config = super(Interpolation, self).get_config()
        config.update({"target_shape": self.target_shape})
        return config

for i, (n_filters, kernel_size) in enumerate(autoencoder_filters):
    conv_block = Conv2D(
        filters= n_filters,
        kernel_size = kernel_size,
        strides = 1,
        padding = 'same',
        kernel_regularizer=l2(regularization_rate),
        kernel_initializer=weightinit
        )(previous_block)
    if IMPROVE:
        conv_block = InstanceNormalization()(conv_block)
        # TODO: Understand better what the PReLU and shared axes are
        conv_block = PReLU(shared_axes=[1])(conv_block)
        conv_block = Dropout(rate = 0.2)(conv_block)
    if i < len(filters):
        # conv_block = MaxPooling2D(pool_size = (2, 1))(conv_block)
        conv_block = MaxPooling2D(pool_size = (2,2))(conv_block)
        shapes.append(conv_block.shape)
    else:
        # conv_block = UpSampling2D(size = (2, 1))(conv_block)
        conv_block = UpSampling2D(size = (2,2))(conv_block)
        opposite_shape = shapes[-(i-len(filters)+2)]
        # if i == len(filters):
        if conv_block.shape[1:3] != opposite_shape[1:3]:
            print(f"{i} {i-len(filters)} {conv_block.shape} != {opposite_shape}")
            conv_block = Interpolation(target_shape = opposite_shape[1:3])(conv_block)
            # conv_block = Lambda(
            #     lambda x: tf.image.resize(x, (x.shape[1] + 1, x.shape[2] + 1))
            #     )(conv_block)
            print(conv_block.shape)
    if i+1 == len(filters):
        encoded = conv_block


    previous_block = conv_block

decoded = Conv2D(
    filters= 1,
    # kernel_size = (5, 1), # TODO :Consider different or variable kernel size
    kernel_size = (3, 3),
    strides = 1,
    activation = "sigmoid",
    padding = 'same',
    kernel_regularizer=l2(regularization_rate),
    kernel_initializer=weightinit
    )(previous_block)

decoded = Reshape(target_shape=shapes[0][1:3])(decoded)

encoder = Model(inputs, encoded)
autoencoder = Model(inputs, decoded)

autoencoder.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=learning_rate),
                    metrics=metrics)

autoencoder.summary()


# In[Train]

earlystopper = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    verbose=1
    )
checkpointer = ModelCheckpoint(
    filepath = output_file,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True
    )

autoencoder.fit(
    x_set_train, x_set_train,
    epochs = 50,
    batch_size = 250,
    shuffle = True,
    validation_data = (x_set_val, x_set_val),
    callbacks = [
        earlystopper,
        checkpointer,
        TensorBoard(log_dir = "/tmp/autoencoder")
        ]
    )
