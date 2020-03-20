#!/usr/bin/env python
# coding: utf-8

# # Try out CNN on averaged EEG data
# 
# ## Pre-processing
# + Import data.
# + Apply filters (bandpass).
# + Detect potential bad channels and replace them by interpolation.
# + Detect potential bad epochs and remove them.
# + Average over a number of randomly drawn epochs (of same person and same stimuli).
# 
# ## Train CNN network
# + Define network architecture
# + Split data
# + Train model
# 

# ## Import packages & links

# In[1]:


# Import packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.getcwd()))

#import mne
#%matplotlib inline
#from mayavi import mlab


# In[2]:


from config import PATH_CODE, PATH_DATA


# ## Load pre-processed dataset
# + See notebook for preprocessing: ePODIUM_prepare_data_for_ML.ipynb.ipynb

# In[3]:

# n_samples = 200
n_samples = 1000

PATH_DATA_processed = os.path.join(PATH_DATA, "test_data")

x_data = np.load(os.path.join(PATH_DATA_processed, f"x_data_{n_samples}.npy"))
y_data = np.load(os.path.join(PATH_DATA_processed, f"y_data_{n_samples}.npy"))


# In[6]:


label_collection = np.unique(y_data)


# In[Separate data by labels]

label_ids_dict = dict()
for label in label_collection:
    label_ids_dict[label] = list()

for i in range(len(y_data)):
    label = y_data[i]
    label_ids_dict[label] = label_ids_dict[label] + [i]


# In[30]:


np.random.seed(1098)
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
    print("Split dataset for label", label, "into train/val/test fractions:", n_train, n_val, n_test)
    
    # Select training, validation, and test IDs:
    trainIDs = np.random.choice(indices, n_train, replace=False)
    valIDs = np.random.choice(list(set(indices) - set(trainIDs)), n_val, replace=False)
    testIDs = list(set(indices) - set(trainIDs) - set(valIDs))
    
    ids_train.extend(list(trainIDs))
    ids_val.extend(list(valIDs))
    ids_test.extend(list(testIDs))


# In[Randomize ids]:

# ids_train = np.array(ids_train)[np.random.permutation(len(ids_train))]
# ids_val = np.array(ids_train)[np.random.permutation(len(ids_val))]
# ids_test = np.array(ids_train)[np.random.permutation(len(ids_test))]

np.random.shuffle(ids_train)
np.random.shuffle(ids_val)
np.random.shuffle(ids_test)


# In[31]:


print(ids_test)


# In[32]:


print(ids_train)



# In[20]:

def binarize_labels(label_dict):
    values = [int(x) for x in label_dict.values()]
    size = np.max(values)
    binarizer_dict = dict()
    for value in set(label_dict.values()):
        bin_value = [0] * (size+1)
        bin_value[int(value)] = 1
        binarizer_dict[value] = bin_value
    return binarizer_dict


label_dict  = {
    'Test_0': '0',
    'Test_1': '1',
    'Test_2': '2',
    'Test_3': '3'
}


binarizer_dict  = binarize_labels(label_dict)


# In[Binarize labels]:

binary_y_data = list()
for label in y_data:
    binary_label = binarizer_dict[label_dict[label]]
    binary_y_data.append(binary_label)



# In[Fake data]

from fake_dataset_generator import FakeDataGenerator

def prepare_sets(indices, full_x_data, full_y_data):
    x_set = list()
    y_set = list()
    for idx in indices:
        x_set.append(full_x_data[idx])
        y_set.append(full_y_data[idx])
    x_set = np.array(x_set)
    x_set = x_set.reshape(np.concatenate((x_set.shape, [1])))
    y_set = np.array(y_set)
    return x_set, y_set

x_set_train, y_set_train = prepare_sets(ids_train, x_data, binary_y_data)
x_set_val, y_set_val = prepare_sets(ids_val, x_data, binary_y_data)
x_set_test, y_set_test = prepare_sets(ids_test, x_data, binary_y_data)

train_generator = FakeDataGenerator(x_set_train, y_set_train)
val_generator = FakeDataGenerator(x_set_val, y_set_val)


# In[34]:


X, y  = train_generator.__getitem__(0)


# In[35]:


print(X.shape)
print(len(y))


# In[36]:


print(y[:11])


# In[42]:


for i in range(min(10, len(y))):
    label = np.where(y[i] == 1)[0][0]
    plt.plot(X[i,:,22], alpha = 0.5, color=(label/5,0, label/5))


# In[ ]:

n_channels = x_data.shape[1]
n_timepoints = x_data.shape[2]


# In[ ]:

for i in range(4):
    fig = plt.figure()
    print(y[i])
    plt.imshow(np.repeat(X[i].reshape((X[i].shape[0:2])), 16, axis = 0))
    fig.show()
    


# In[26]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[Disable eager execution]:

tf.compat.v1.disable_eager_execution() # TODO Delete, substitute gradients for GradientTape


# In[Define model functions]

def start_training(
        model, output_file, train_generator, val_generator, epochs = 50):
    checkpointer = ModelCheckpoint(
        filepath = output_file, 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True
        )
    earlystopper = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=1
        )
    
    model.fit(
        x=train_generator, 
        validation_data=val_generator,
        epochs=epochs,
        callbacks = [
            checkpointer,
            earlystopper,
            ]
        )


def load_model(model, output_file):
    if os.path.isfile(output_file):
        try:
            model.load_weights(output_file)
            print(f"Loaded weights from {output_file}")
        except Exception as e:
            print(repr(e))


def compile_model(model):
    #Adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    #model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[Clear memory]
tf.keras.backend.clear_session()
try:
    del model
except:
    pass

# In[Define model to use]
# reduce_on_time = True
reduce_on_time = False

# In[40]:


# Simple CNN model
input_shape = X.shape[1:]
n_outputs = len(label_collection)
print(f"Input shape: {input_shape}; n_outputs: {n_outputs}")

model = tf.keras.Sequential()
#model.add(layers.Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(layers.Conv2D(filters=32, kernel_size=(1, 20), input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2D(filters=64, kernel_size=(1, 10))) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

if reduce_on_time:
    model.add(layers.Conv2D(filters=32, kernel_size=(1, 5))) #, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.AveragePooling2D(pool_size=(1, 46))) # Reducing dimensionality on time dimension
else:
    model.add(layers.Conv2D(filters=4, kernel_size=(1, 1))) # Reducing dimensionality on filters dimension
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.AveragePooling2D(pool_size=(1, 4))) 

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_outputs, activation='softmax'))


# In[39]:


model.summary()


# In[Output file]:

# Save best model and include early stopping
if reduce_on_time:
    output_filename = 'Test_data_classifier_avg_pool-100.hdf5'
else:
    output_filename = 'Test_data_classifier_avg_pool-4_conv-1-4.hdf5'
output_file = os.path.join(PATH_CODE, 'models_trained' , output_filename)


# In[Load model]:

if do_load:
    load_model(model, output_file)


# In[Compile]:

compile_model(model)



# In[Train]:
if do_train:
    start_training(model, output_file, train_generator, val_generator)

# In[Define Visualize Grad_Cam]
from matplotlib import pyplot as plt

def visualize_gradcam(
        gradcam,
        network_input = None,
        label = None,
        layer = None):
    fig = plt.figure(figsize = (16, 6.4))
    plt.axis('off')
    ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    n_plots = 2
    n_halves = 0
    title = ""
    if label is not None:
        title += f", class {label}"
    if layer is not None:
        title += f", layer {layer}"
    plt.title(f'GradCAM{title}')
    if network_input is not None:
        n_halves = 1
        fig.add_subplot(2, 1, 1)
        # fig.add_subplot(1, 2, 1)
        plt.axis('off')
        # plt.imshow(np.transpose(network_input))
        # plt.imshow(network_input.reshape(network_input.shape[0:2]))
        plt.imshow(np.repeat(network_input.reshape(network_input.shape[0:2]), 8, 0))
        plt.colorbar(ticks=ticks, orientation='horizontal')
        
    gradcam_floored = np.maximum(gradcam, 0)

    # fig.add_subplot(n_halves + 1, n_plots, n_halves*n_plots +1)
    fig.add_subplot(n_plots, n_halves + 1, n_halves*n_plots +1)
    plt.axis('off')
    plt.imshow(np.repeat(gradcam_floored, 16, 0))
    plt.colorbar(ticks=ticks, orientation='horizontal')
    
    # fig.add_subplot(n_halves +1, n_plots, n_halves*n_plots +2)
    fig.add_subplot(n_plots, n_halves +1, n_halves*n_plots +2)
    plt.axis('off')
    plt.imshow(np.repeat(gradcam, 16, 0))
    plt.colorbar(ticks=ticks, orientation='horizontal')
    # fig.show()

# In[GradCam]

from grad_cam import grad_cam

input_model = model
for i in range(4):
    input_image = x_set_val[i]
    label = y_set_val[i]
    layer_name = "conv2d_2"
    # layer_name = "average_pooling2d"
    
    gradcam = grad_cam(input_model, input_image, 0, layer_name)
    visualize_gradcam(gradcam, input_image, label = label)
    gradcam = grad_cam(input_model, input_image, 1, layer_name)
    visualize_gradcam(gradcam, input_image, label = label)
    gradcam = grad_cam(input_model, input_image, 2, layer_name)
    visualize_gradcam(gradcam, input_image, label = label)
    gradcam = grad_cam(input_model, input_image, 3, layer_name)
    visualize_gradcam(gradcam, input_image, label = label)


# In[Clear memory]
tf.keras.backend.clear_session()
del model


# In[Model based on assafExplainableDeep2019]:


model = tf.keras.Sequential()
# k timepoints * 1 channel
model.add(layers.Conv2D(filters=32, kernel_size = (1, 20), input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

# 1 * 1 Convolution to reduce feature maps
model.add(layers.Conv2D(filters=1, kernel_size=(1, 1)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
# End of first stage

# 1D Convolution, k*#channels
model.add(layers.Conv2D(filters=32, kernel_size = (input_shape[0], 10)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

# End of second stage
model.add(layers.Conv2D(filters=4, kernel_size=(1, 1))) # Reducing dimensionality on filters dimension
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling2D(pool_size=(1, 4))) 


model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_outputs, activation='softmax'))

model.summary()


# In[Output file]:

# Save best model and include early stopping

output_filename = f'Test_data_assaf2019_avg_pool-4_conv-1-4.hdf5'
output_file = os.path.join(PATH_CODE, 'models_trained' , output_filename)


# In[Load model]:

if do_load:
    load_model(model, output_file)


# In[Compile]:

compile_model(model)


# In[Train]:
if do_train:
    start_training(model, output_file, train_generator, val_generator)

# In[Channels version]

"""Model from Assaf 2019 modified to give results about channels"""

# In[Clear memory]
tf.keras.backend.clear_session()
try:
    del model
except:
    pass


# In[Model based on assafExplainableDeep2019, ]:


model = tf.keras.Sequential()
# k timepoints * 1 channel
model.add(layers.Conv2D(filters=32, kernel_size = (1, 20), input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

# 1 * 1 Convolution to reduce feature maps
model.add(layers.Conv2D(filters=1, kernel_size=(1, 1)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
# End of first stage

# 1D Convolution, k*#channels
model.add(layers.Conv2D(filters=32, kernel_size = (1, 381)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

# End of second stage
model.add(layers.Conv2D(filters=4, kernel_size=(1, 1))) # Reducing dimensionality on filters dimension
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling2D(pool_size=(1, 1))) 


model.add(layers.Flatten())
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_outputs, activation='softmax'))

model.summary()


# In[Output file]:

# Save best model and include early stopping

output_filename = f'Test_data_assaf2019_avg_pool-4_conv-1-4-channels.hdf5'
output_file = os.path.join(PATH_CODE, 'models_trained' , output_filename)


# In[Load model]:

if do_load:
    load_model(model, output_file)


# In[Compile]:

compile_model(model)


# In[Train]:
if do_train:
    start_training(
        model, output_file, train_generator, val_generator, epochs = 200)


# In[GradCam]

from grad_cam import grad_cam

input_model = model
# for layer in input_model.layers[:-4]:
#     idx_input = 5
for idx_input in [5, 0, 1, 4]:
    layer = input_model.layers[-5]
    input_image = x_set_val[idx_input]
    true_label = y_set_val[idx_input]
    # layer_name = "conv2d_1"
    # layer_name = "leaky_re_lu_1"
    # layer_name = "conv2d_3"
    layer_name = layer.name
    print(layer_name)

    grad_cam_label = np.argmax(model.predict([[input_image]]))
    print(f"Predicted label: {binarizer_dict[str(grad_cam_label)]}, True label: {true_label}")
    
    gradcam = grad_cam(input_model, input_image, grad_cam_label, layer_name)
    visualize_gradcam(
        gradcam, input_image, label = true_label, layer = layer_name)
    
    # gradcam = grad_cam(input_model, input_image, 0, layer_name)
    # visualize_gradcam(gradcam, input_image, label = label, layer = layer_name)
    # gradcam = grad_cam(input_model, input_image, 1, layer_name)
    # visualize_gradcam(gradcam, input_image, label = label, layer = layer_name)
    # gradcam = grad_cam(input_model, input_image, 2, layer_name)
    # visualize_gradcam(gradcam, input_image, label = label, layer = layer_name)
    # gradcam = grad_cam(input_model, input_image, 3, layer_name)
    # visualize_gradcam(gradcam, input_image, label = label, layer = layer_name)


# In[Define confusion matrix]:

from sklearn import metrics

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # from sklearn.utils.multiclass import unique_labels
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    ## Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[Plot confusion matrix]

labels = list(binarizer_dict.keys())

true_labels = y_set_test
predicted_labels = model.predict(x_set_test)


plot_confusion_matrix(np.argmax(true_labels, axis=1), 
                      np.argmax(predicted_labels, axis=1), 
                      classes=labels, normalize=True,
                      title='Normalized confusion matrix')





# In[Eli5 visualization (low level)]:

from eli5.keras import gradcam as eli5_gc

estimator = model
doc = input_image
activation_layer = layer_name

weights, activations, gradients, predicted_idx, predicted_val \
    = eli5_gc.gradcam_backend(estimator, doc, None, activation_layer)

eli5_gradcam = eli5_gc.gradcam(weights, activations)
visualize_gradcam(eli5_gradcam)