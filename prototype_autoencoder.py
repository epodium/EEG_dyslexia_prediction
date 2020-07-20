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
    UpSampling2D, Reshape, PReLU, Dropout, Lambda, Layer, Flatten, \
    Conv2DTranspose, BatchNormalization, ReLU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from real_data_utils import prepare_set, exchange_channels
import benchmark_data_utils


# In[Configure experiment]

from config import PATH_CODE, PATH_DATA, ROOT

autoencoder_model_name = "prototype_autoencoder"
classifier_model_name = "prototype_classifier"

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


# do_load = True; do_train = False
do_load = False; do_train = True


# data_type = "mnist"
data_type = "benchmark"

if do_load:
    # timestamp = "20200703-110654"
    timestamp = "20200701-165745"

if data_type == "real":
    dataset_folder = 'processed_data_17mn'
    dataset_folder = 'processed_data_29mnd'
    # dataset_folder = "processed_data_41mnd"
elif data_type == "benchmark":

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
if data_type == "real":
    pass # TODO
elif data_type == "mnist":
    from tensorflow.keras.datasets import mnist
    (x_set_train, y_set_train), (x_set_val, y_set_val) = mnist.load_data()
    x_set_train = x_set_train.astype('float32') / 255.
    x_set_val = x_set_val.astype('float32') / 255.

    # x_set_train = np.reshape(x_set_train, (len(x_set_train), 28, 28, 1))
    # x_set_val = np.reshape(x_set_val, (len(x_set_val), 28, 28, 1))
    # x_set_train = tf.image.resize(x_set_train, (32, 32))
    # x_set_val = tf.image.resize(x_set_val, (32, 32))
    # x_set_train = np.reshape(x_set_train, (len(x_set_train), 32, 32))
    # x_set_val = np.reshape(x_set_val, (len(x_set_val), 32, 32))

    input_shape = x_set_train.shape

elif data_type == "benchmark":
    x_data, y_data, data_type = benchmark_data_utils.load_data(
        PATH_DATA_processed,
        ts_type,
        n_samples,
        ignore_noise)

    # Scale data to the range 0-1
    x_data_max = np.max(x_data)
    x_data_min = np.min(x_data)
    x_data = (x_data-x_data_min) / (x_data_max - x_data_min)

    # Separate data by labels
    label_collection, label_ids_dict = benchmark_data_utils.collect_labels(y_data)

    # Transform labels to integer
    label_dict = dict()
    for i, label in enumerate(label_collection):
        label_dict[label] = i
    y_data_int = np.array([label_dict[label] for label in y_data])

    # Split labels
    ids_train, ids_val, ids_test = benchmark_data_utils.split_labels(label_collection, label_ids_dict, 1098)


    # Randomize ids
    np.random.shuffle(ids_train)
    np.random.shuffle(ids_val)
    np.random.shuffle(ids_test)


    # Split datasets
    # NOTE: mcfly takes data in the shape of length timeseries and channels

    print(x_data.shape)
    x_data = exchange_channels(x_data)
    print(x_data.shape)

    x_set_train = x_data[ids_train]
    y_set_train = y_data_int[ids_train]
    x_set_val = x_data[ids_val]
    y_set_val = y_data_int[ids_val]
    # x_set_test = x_data[ids_test]

    input_shape = x_set_train.shape


# In[Configure autoencoder]

# IMPROVE = False
# # IMPROVE = True
dim_length = input_shape[1]  # number of samples in a time series
dim_channels = input_shape[2]  # number of channels
weightinit = 'lecun_uniform'  # weight initialization
regularization_rate = 10 ** -2
learning_rate = 10 ** -4
metrics = ["accuracy"]
batch_size = 250
dense_layer = False
# dense_layer = True

autoencoder_filename = f"{autoencoder_model_name}-{data_type}-{timestamp}"
autoencoder_output_file = os.path.join(PATH_CODE,
                                       'models_trained',
                                       "autoencoder" ,
                                       f"{autoencoder_filename}.hdf5")




# In[Define autoencoder]

tf.keras.backend.clear_session()

inputs = Input(shape = (dim_length, dim_channels))
reshape = Reshape(target_shape=(dim_length, dim_channels, 1))(inputs)

previous_block = reshape
# filters = [(128, 5), (256, 11), (512, 21)]
# filters = [(16, 5), (32, 11), (64, 21)]
# filters = [(16, (3, 3)), (8, (3, 3)), (8, (3, 3))]
if data_type == "mnist":
    filters = [(16, (3, 3)), (8, (3, 3))]
# filters = [(32, (5, 1)), (16, (11, 1)), (8, (21, 1))]
# filters = [(32, (11, 1)), (16, (11, 1)), (8, (11, 1))]
if data_type == "benchmark1-noise1":
    filters = [(1, (2, 1)), (32, (2, 1)), (16, (3, 1)), (8, (3, 1))]
    # filters = [(32, (5, 1)), (16, (5, 1)), (8, (5, 1)), (4, (5, 1))]
    # filters = [(32, (5, 1)), (16, (11, 1)), (8, (21, 1)), (4, (41, 1))]
    if dense_layer:
        filters += [(4, (3, 1))]
        # filters += [(2, (5, 1))]
        # filters += [(2, (81, 1))]
    else:
        filters += [(1, (3, 1))]
        # filters += [(1, (5, 1))]
        # filters += [(1, (81, 1))]
pool_size = (2, 1)
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
    if i < len(filters):
        conv_block = Conv2D(
            filters= n_filters,
            kernel_size = kernel_size,
            strides = 1,
            padding = 'same',
            # activation='relu', # From Tutorial
            kernel_regularizer=l2(regularization_rate),
            kernel_initializer=weightinit
            )(previous_block)
        conv_block = BatchNormalization()(conv_block)
        conv_block = MaxPooling2D(
            # padding = 'same', # From Tutorial
            pool_size = pool_size
            )(conv_block)
        conv_block = ReLU()(conv_block)
        shapes.append(conv_block.shape)
    else:
        conv_block = Conv2DTranspose(
            filters= n_filters,
            kernel_size = kernel_size,
            strides = 1,
            padding = 'same',
            # activation='relu', # From Tutorial
            kernel_regularizer=l2(regularization_rate),
            kernel_initializer=weightinit
            )(previous_block)
        conv_block = UpSampling2D(size = pool_size)(conv_block)
        conv_block = ReLU()(conv_block)
        conv_block = BatchNormalization()(conv_block)
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
        if dense_layer:
            shape = conv_block.shape
            conv_block = Flatten()(conv_block)
            conv_block = Dense(100)(conv_block)
        encoded = conv_block
        if dense_layer:
            conv_block = Dense(shape[1]*shape[2]*shape[3])(conv_block)
            conv_block = Reshape(target_shape = shape[1:])(conv_block)
    previous_block = conv_block

decoded = previous_block # First convolution is already at 1 filter
# decoded = Conv2D( # First convolution is already at 1 filter
#     filters= 1,
#     # kernel_size = (5, 1), # TODO :Consider different or variable kernel size
#     kernel_size = (1, 1), # TODO :Consider different or variable kernel size
#     # kernel_size = (3, 3),
#     strides = 1,
#     # activation = "sigmoid", # From tutorial
#     padding = 'same',
#     kernel_regularizer=l2(regularization_rate),
#     kernel_initializer=weightinit
#     )(previous_block)

decoded = Reshape(target_shape=shapes[0][1:3])(decoded)

autoencoder = Model(inputs, decoded)

autoencoder.summary()


# In[Train autoencoder]

patience_autoencoder = 5

if do_load:
    try:
        autoencoder.load_weights(autoencoder_output_file)
        print(f"Loaded weights from {autoencoder_output_file}")
    except Exception as e:
        print(repr(e))

autoencoder.compile(
    # loss='mse',
    loss='binary_crossentropy',
    # loss = 'huber_loss',
    optimizer=Adam(lr=learning_rate),
    metrics=metrics)

# # From Tutorial
# autoencoder.compile(optimizer='adadelta',
#                     loss='binary_crossentropy',
#                     metrics=metrics)




if do_train:

    checkpointer = ModelCheckpoint(
        filepath = autoencoder_output_file,
        # monitor='val_accuracy',
        monitor='val_loss', mode="min",
        verbose=1,
        save_best_only=True
        )

    earlystopper_autoencoder = EarlyStopping(
        # monitor='val_accuracy',
        monitor='val_loss', mode="min",
        patience=patience_autoencoder,
        verbose=1
        )

    autoencoder.fit(
        x_set_train, x_set_train,
        epochs = 50,
        batch_size = batch_size,
        shuffle = True,
        validation_data = (x_set_val, x_set_val),
        callbacks = [
            earlystopper_autoencoder,
            checkpointer,
            TensorBoard(log_dir = f"/tmp/tensorboard/{timestamp}-{data_type}/autoencoder/")
            ]
        )


# In[Visually inspect results]

decoded_imgs = autoencoder.predict(x_set_val)
label_imgs = y_set_val

# In[Visualize reconstructions]
import matplotlib.pyplot as plt
from matplotlib import cm

plt.style.use('ggplot')

def plot_comparison(original, reconstruction, ax = None):
    n_points, n_ch = original.shape[0:2]

    # bg_cmap = cm.get_cmap('inferno')

    if ax == None:
        fig, ax = plt.subplots(figsize=(20,(1+0.7 *n_ch*2)))

    for i in range(n_ch):
        ax.plot(original[:, i] - i, color = "black")
        ax.plot(reconstruction[:, i] - i, color = "red")
    ax.set_yticks(-np.arange(n_ch))
    ax.set_yticklabels(['channel ' + str(i) for i in range(n_ch)])


def plot_difference(original, reconstruction, ax = None):
    n_points, n_ch = original.shape[0:2]
    # if ax == None:
    #     fig, ax = plt.subplots(figsize=(20,(1+0.7 *n_ch*2)))
    for i in range(n_ch):
        ax.plot(original[:, i] - reconstruction[:, i] - i, color = "blue")
    ax.set_yticks(-np.arange(n_ch))
    ax.set_yticklabels(['channel ' + str(i) for i in range(n_ch)])


n_figs = 6 # Total number of figures
n_plots = 2 # Number of plots per figure
figsize = (20, 16)
for i_fig in range(n_figs):
    fig_comparison = plt.figure(figsize=figsize)
    fig_difference = plt.figure(figsize=figsize)
    for i in range(n_plots):
        item = i+ 6*n_plots*i_fig
        n_rows = 1
        n_cols = 2

        original = x_set_val[item]
        reconstruction = decoded_imgs[item]
        label = label_imgs[item]

        ax_comparison = fig_comparison.add_subplot(n_rows, n_cols, i+1)
        ax_difference = fig_difference.add_subplot(n_rows, n_cols, i+1)

        axes = [ax_comparison, ax_difference]
        for ax in axes:
            ax.get_xaxis().set_visible(False)
            # a.get_yaxis().set_visible(False)
            ax.set_title(label)

        # ax_difference.get_xaxis().set_visible(False)
        # ax_difference.get_yaxis().set_visible(False)
        # ax_difference.set_title(label)

        plot_comparison(original, reconstruction, ax_comparison)
        plot_difference(original, reconstruction, ax_difference)
    plt.draw()


# In[Encode data for classification]

print("Encoding data")
encoder = Model(inputs, encoded)
encoded_x_train = encoder.predict(x_set_train)
encoded_x_val = encoder.predict(x_set_val)
input_shape = encoded_x_train.shape[1:]

from tensorflow.keras.utils import to_categorical
n_outputs = len(np.unique(y_set_train))
cat_y_train = to_categorical(y_set_train, n_outputs)
cat_y_val = to_categorical(y_set_val, n_outputs)


# In[Print encoded data]
if not dense_layer:
    n = 10
    plt.figure(figsize=(10, 10))
    for i in range(n):
        # i += 4*n

        # display encoded data
        ax = plt.subplot(2, n, i+1)
        plt.imshow(encoded_x_val[i][:,:,0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # # display encoded data
        # ax = plt.subplot(2, n, i+n+1)
        # plt.imshow(encoded_x_val[i][:,:,1])
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    plt.draw()


# In[Define classifier]
model = tf.keras.Sequential()
model.add(Flatten(input_shape = input_shape))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_outputs, activation='softmax'))

patience_classifier = 50

# In[Train classifier]

classifier_filename = f"{classifier_model_name}-{data_type}-{timestamp}"
classifier_output_file = os.path.join(PATH_CODE,
                                       'models_trained',
                                       "autoencoder" ,
                                       f"{classifier_filename}.hdf5")

if do_load:
    try:
        model.load_weights(classifier_output_file)
        print(f"Loaded weights from {classifier_output_file}")
    except Exception as e:
        print(repr(e))

model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=learning_rate),
                    metrics=metrics)

if do_train:
    checkpointer = ModelCheckpoint(
        filepath = classifier_output_file,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True
        )

    earlystopper_classifier = EarlyStopping(
        monitor='val_accuracy',
        patience=patience_classifier,
        verbose=1
        )

    model.fit(
            encoded_x_train, cat_y_train,
            epochs = 1000,
            batch_size = batch_size,
            shuffle = True,
            validation_data = (encoded_x_val, cat_y_val),
            callbacks = [
                earlystopper_classifier,
                checkpointer,
                TensorBoard(log_dir = f"/tmp/tensorboard/{timestamp}-{data_type}/classifier")
                ]
            )


# In[Clustering]
print("Clustering")

cluster_type = "train"
# cluster_type = "val"

if cluster_type == "train":
    data_labels = y_set_train
    encoded_data = encoded_x_train
elif cluster_type == "val":
    data_labels = y_set_val
    encoded_data = encoded_x_val


from sklearn.manifold import TSNE
if dense_layer:
    tsne_data = encoded_data
else:
    data_shape = encoded_data.shape
    tsne_data = encoded_data.reshape(
        data_shape[0],
        data_shape[1] * data_shape[2] * data_shape[3])
data_embedded = TSNE(n_jobs = -1).fit_transform(tsne_data)


# In[Plot cluster]
fig, ax = plt.subplots(figsize=(10, 10))
scatter= ax.scatter(
        data_embedded.T[0],
        data_embedded.T[1],
        c = data_labels,
        s = 20,
        # cmap = 'gist_rainbow')
        cmap = 'nipy_spectral')
fig.legend(*scatter.legend_elements())
plt.show()
