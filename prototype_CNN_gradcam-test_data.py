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


PATH_DATA_processed = os.path.join(PATH_DATA, "test_data")

x_data = np.load(os.path.join(PATH_DATA_processed, "x_data.npy"))
y_data = np.load(os.path.join(PATH_DATA_processed, "y_data.npy"))


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

def prepare_generator(indices, full_x_data, full_y_data):
    x_set = list()
    y_set = list()
    for idx in indices:
        x_set.append(full_x_data[i])
        y_set.append(full_y_data[i])
    x_set = np.array(x_set)
    x_set = x_set.reshape(np.concatenate((x_set.shape, [1])))
    y_set = np.array(y_set)
    return x_set, y_set

x_set_train, y_set_train = prepare_generator(ids_train, x_data, binary_y_data)
x_set_val, y_set_val = prepare_generator(ids_val, x_data, binary_y_data)
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
    plt.imshow(X[i].reshape((X[i].shape[0:2])))
    fig.show()


# In[26]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[Disable eager execution]:

tf.compat.v1.disable_eager_execution() # TODO Delete, substitute gradients for GradientTape


# In[Define model functions]

def start_training(model, output_file, train_generator, val_generator):
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
        epochs=50,
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
del model


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

model.add(layers.Conv2D(filters=32, kernel_size=(1, 5))) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling2D(pool_size=(1, 46))) # Reducing dimensionality on time dimension
#model.add(layers.GlobalAveragePooling1D(data_format=None))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_outputs, activation='softmax'))


# In[39]:


model.summary()


# In[Output file]:

# Save best model and include early stopping
output_filename = 'Test_data_classifier_avg_pool-100.hdf5'
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

def visualize_gradcam(gradcam, network_input = None):
    fig = plt.figure(figsize = (8, 6.4))
    ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    n_plots = 2
    first_plot = 0
    if network_input is not None:
        n_plots = 3
        first_plot = 1
        fig.add_subplot(n_plots, 1, 1)
        plt.axis('off')
        # plt.imshow(np.transpose(network_input))
        plt.imshow(network_input.reshape(network_input.shape[0:2]))
        plt.colorbar(ticks=ticks, orientation='horizontal')
    # plt.title('GradCAM')
    fig.add_subplot(n_plots, 1, first_plot +1)
    plt.axis('off')
    plt.imshow(np.maximum(gradcam, 0))
    plt.colorbar(ticks=ticks, orientation='horizontal')
    fig.show()
    
    fig.add_subplot(n_plots, 1, first_plot +2)
    plt.axis('off')
    plt.imshow(gradcam)
    plt.colorbar(ticks=ticks, orientation='horizontal')
    fig.show()

# In[GradCam]

from grad_cam import grad_cam

input_model = model
for i in range(4):
    input_image = x_set_val[i]
    # layer_name = "conv2d_2"
    layer_name = "average_pooling2d"
    
    gradcam = grad_cam(input_model, input_image, 0, layer_name)
    visualize_gradcam(gradcam, input_image)
    gradcam = grad_cam(input_model, input_image, 1, layer_name)
    visualize_gradcam(gradcam, input_image)
    gradcam = grad_cam(input_model, input_image, 2, layer_name)
    visualize_gradcam(gradcam, input_image)
    gradcam = grad_cam(input_model, input_image, 3, layer_name)
    visualize_gradcam(gradcam, input_image)




# In[81]:


model.get_weights()


# In[Clear memory]
tf.keras.backend.clear_session()
del model


# In[Network reduced dimensionality with Conv 1x1]

# Simple CNN model

model = tf.keras.Sequential()
#model.add(layers.Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(layers.Conv2D(filters=32, kernel_size=(1, 20), input_shape=(input_shape)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2D(filters=64, kernel_size=(1, 10))) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2D(filters=32, kernel_size=(1, 5))) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2D(filters=4, kernel_size=(1, 1))) # Reducing dimensionality on filters dimension
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.AveragePooling2D(pool_size=(1, 4))) 
#model.add(layers.GlobalAveragePooling1D(data_format=None))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_outputs, activation='softmax'))


# In[39]:


model.summary()


# In[Output file]:

# Save best model and include early stopping
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

def visualize_gradcam(gradcam):
    plt.figure(figsize=(15, 10))
    plt.title('GradCAM')
    plt.axis('off')
    #plt.imshow(np.transpose(network_input))
    im = plt.imshow(np.transpose(gradcam), cmap='jet', alpha=0.5)
    im.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.show()


# In[GradCam]

from grad_cam import grad_cam

input_model = model
i = 3
input_image = X[i]
n_class = np.argmax(model.predict(np.array([X[i]])))
# n_class = np.argmin(model.predict(np.array([X[i]])))
# layer_name = "conv2d_3"
layer_name = "average_pooling2d"

gradcam = grad_cam(input_model, input_image, n_class, layer_name)


# In[Visualize]
visualize_gradcam(gradcam.transpose())


# In[Eli5 visualization (high level)]:
import eli5
from IPython.display import display


expl = eli5.explain_prediction(model, input_image)
heatmap = expl.targets[0].heatmap
heatmap_im = eli5.formatters.image.heatmap_to_image(heatmap)
display(heatmap_im)



# In[Eli5 visualization (low level)]:

from eli5.keras import gradcam as eli5_gc

estimator = model
doc = input_image
activation_layer = layer_name

weights, activations, gradients, predicted_idx, predicted_val \
    = eli5_gc.gradcam_backend(estimator, doc, None, activation_layer)

eli5_gradcam = eli5_gc.gradcam(weights, activations)
visualize_gradcam(eli5_gradcam)