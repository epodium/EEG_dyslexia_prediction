#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:07 2020

@author: Breixo Solino
"""

# In[Import packages]

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, \
    UpSampling2D, Reshape, PReLU, Dropout
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam



# In[Acquire data]


# In[Configure autoencoder]

IMPROVE = False
dim_length = x_shape[1]  # number of samples in a time series
dim_channels = x_shape[2]  # number of channels
outputdim = number_of_classes  # number of classes
weightinit = 'lecun_uniform'  # weight initialization
regularization_rate = 10 ** -2
learning_rate = 10 ** -4
metrics = "accuracy"


# In[Define autoencoder]

inputs = Input(shape = (dim_length, dim_channels))
reshape = Reshape(target_shape=(dim_length, dim_channels, 1))(inputs)

previous_block = reshape
filters = [(128, 5), (256, 11), (512, 21)]
autoencoder_filters = filters + list(reversed(filters))

for i, (n_filters, kernel_size) in enumerate(autoencoder_filters):
    conv_block = Conv2D(
        filters= n_filters,
        kernel_size = (kernel_size, 1), # TODO :Consider different or variable kernel size
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
        conv_block = MaxPooling2D(pool_size = (2, 1))(conv_block)
    else:
        conv_block = UpSampling2D(pool_size = (2, 1))(conv_block)
    if i+1 == len(filters):
        encoded = conv_block
    previous_block = conv_block

decoded = conv_block = Conv2D(
    filters= 1,
    kernel_size = (5, 1), # TODO :Consider different or variable kernel size
    strides = 1,
    activation = "sigmoid",
    padding = 'same',
    kernel_regularizer=l2(regularization_rate),
    kernel_initializer=weightinit
    )(previous_block)

encoder = Model(inputs, encoded)
autoencoder = Model(inputs, decoded)

autoencoder.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=learning_rate),
                    metrics=metrics)

