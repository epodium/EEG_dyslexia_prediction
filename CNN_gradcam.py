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
import csv
import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.getcwd()))

#import mne
#%matplotlib inline
#from mayavi import mlab


# In[2]:


from config import ROOT, PATH_CODE, PATH_DATA, PATH_OUTPUT, PATH_METADATA
PATH_CNTS = os.path.join(PATH_DATA, "17mnd mmn")


# ## Load pre-processed dataset
# + See notebook for preprocessing: ePODIUM_prepare_data_for_ML.ipynb.ipynb

# In[3]:


PATH_DATA_processed = os.path.join(PATH_DATA, 'processed_data_17mn')

dirs = os.listdir(PATH_DATA_processed)
files_npy = fnmatch.filter(dirs, "*.npy")
files_csv = fnmatch.filter(dirs, "*.csv")


# In[4]:


print(len(files_csv))
print(len(files_npy))


# ### Count the (main) labels for all present files

# In[5]:


def read_labels(filename, PATH):
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


# In[6]:


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


# In[7]:


def transform_label(label,
                   label_dict,
                   main_label_dict = None):
    if label in label_dict:
        label_new = label_dict[label]
    else:
        label_new = None
    
    if main_label_dict is not None:
        if label in main_label_dict:
            main_label = main_label_dict[label]
        else:
            main_label = None
        return label_new, main_label
    
    else:
        return label_new
    


# ### Define new labels

# In[8]:


label_dict  = {
    '3dys0_risk0': '0',
    '13dys0_risk0': '1',
    '66dys0_risk0': '2',
    '3dys0_risk1': '0',
    '13dys0_risk1': '1',
    '66dys0_risk1': '2',
    '3dys1_risk0': '3',
    '13dys1_risk0': '4',
    '66dys1_risk0': '5',
    '3dys1_risk1': '3',
    '13dys1_risk1': '4',
    '66dys1_risk1': '5',
}


main_label_dict  = {
    '3dys0_risk0': '0',
    '13dys0_risk0': '0',
    '66dys0_risk0': '0',
    '3dys0_risk1': '0',
    '13dys0_risk1': '0',
    '66dys0_risk1': '0',
    '3dys1_risk0': '1',
    '13dys1_risk0': '1',
    '66dys1_risk0': '1',
    '3dys1_risk1': '1',
    '13dys1_risk1': '1',
    '66dys1_risk1': '1',
}
"""
main_label_dict  = {
    '3dys0_risk0': '0',
    '13dys0_risk0': '0',
    '66dys0_risk0': '0',
    '3dys0_risk1': '1',
    '13dys0_risk1': '1',
    '66dys0_risk1': '1',
    '3dys1_risk0': '0',
    '13dys1_risk0': '0',
    '66dys1_risk0': '0',
    '3dys1_risk1': '1',
    '13dys1_risk1': '1',
    '66dys1_risk1': '1',
}
"""


# ### Collect main labels (here: dyslexic 0 | 1)

# In[9]:


main_labels = [transform_label(x[0], label_dict, main_label_dict)[1] for x in label_collection]
print(len(main_labels))


# In[10]:


print(main_labels.count('0'))
print(main_labels.count('1'))


# ## Import and initiate data generator function

# In[11]:


from dataset_generator import DataGenerator


# In[12]:


print(files_csv[:5])


# ### Split data set

# In[13]:


for label in list(set(main_labels)):
    print("Found datapoints for label", label, "--->", main_labels.count(label))


# In[14]:


np.where(np.array(main_labels) == '1')[0].shape


# In[30]:


np.random.seed(1098)
split_ratio = (0.7, 0.15, 0.15)

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


# In[31]:


print(IDs_test)


# In[32]:


print(IDs_train)


# In[19]:


label_dict  = {
    '3dys0_risk0': '0',
    '13dys0_risk0': '0',
    '66dys0_risk0': '0',
    '3dys0_risk1': '1',
    '13dys0_risk1': '1',
    '66dys0_risk1': '1',
    '3dys1_risk0': '0',
    '13dys1_risk0': '0',
    '66dys1_risk0': '0',
    '3dys1_risk1': '1',
    '13dys1_risk1': '1',
    '66dys1_risk1': '1',
}


# In[20]:


label_dict  = {
    '3dys0_risk0': '0',
    '13dys0_risk0': '0',
    '66dys0_risk0': '0',
    '3dys0_risk1': '0',
    '13dys0_risk1': '0',
    '66dys0_risk1': '0',
    '3dys1_risk0': '1',
    '13dys1_risk0': '1',
    '66dys1_risk0': '1',
    '3dys1_risk1': '1',
    '13dys1_risk1': '1',
    '66dys1_risk1': '1',
}


# In[20]:


label_dict  = {
    '3dys0_risk0': '0',
    '13dys0_risk0': '1',
    '66dys0_risk0': '2',
    '3dys0_risk1': '0',
    '13dys0_risk1': '1',
    '66dys0_risk1': '2',
    '3dys1_risk0': '3',
    '13dys1_risk0': '4',
    '66dys1_risk0': '5',
    '3dys1_risk1': '3',
    '13dys1_risk1': '4',
    '66dys1_risk1': '5',
}


binarizer_dict  = {
    '0': [1, 0, 0, 0, 0, 0],
    '1': [0, 1, 0, 0, 0, 0],
    '2': [0, 0, 1, 0, 0, 0],
    '3': [0, 0, 0, 1, 0, 0],
    '4': [0, 0, 0, 0, 1, 0],
    '5': [0, 0, 0, 0, 0, 1]
}


# In[124]:


# give labels of data that should not be used for training/testing

ignore_labels = ['3dys0_risk0', '3dys0_risk1', '3dys1_risk0', '3dys1_risk1',
                '13dys0_risk0', '13dys0_risk1', '13dys1_risk0', '13dys1_risk1']


# In[33]:


train_generator = DataGenerator(list_IDs = IDs_train,
                                 main_labels = main_labels,
                                 label_dict = label_dict,
                                 binarizer_dict = binarizer_dict,
                                 ignore_labels = ignore_labels,
                                 path_data = PATH_DATA_processed,
                                 filenames = [x[:-4] for x in files_csv],
                                 data_path = PATH_DATA_processed, 
                                 to_fit=True, 
                                 n_average = 40,
                                 batch_size = 10,
                                 iter_per_epoch = 30,
                                 up_sampling = True,
                                 n_timepoints = 501,
                                 n_channels=30, 
#                                 n_classes=1, 
                                 shuffle=True)

val_generator = DataGenerator(list_IDs = IDs_val,
                                 main_labels = main_labels,
                                 label_dict = label_dict,
                                 binarizer_dict = binarizer_dict,
                                 ignore_labels = ignore_labels,
                                 path_data = PATH_DATA_processed,
                                 filenames = [x[:-4] for x in files_csv],
                                 data_path = PATH_DATA_processed, 
                                 to_fit=True, 
                                 n_average = 40,
                                 batch_size = 10,
                                 iter_per_epoch = 30,
                                 up_sampling = True,
                                 n_timepoints = 501,
                                 n_channels=30, 
#                                 n_classes=1, 
                                 shuffle=True)


# In[34]:


X, y  = train_generator.__getitem__(0)


# In[35]:


print(X.shape)
print(len(y))


# In[36]:


print(y[:11])


# In[42]:


for i in range(10):
    label = np.where(y[i] == 1)[0][0]
    plt.plot(X[i,:,22], alpha = 0.5, color=(label/5,0, label/5))


# In[ ]:





# In[ ]:





# ## Define model architecture

# In[26]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[Disable eager execution]:

tf.compat.v1.disable_eager_execution() # TODO Delete, substitute gradients for GradientTape


# In[40]:


# Simple CNN model
n_timesteps = 501
n_features = 30
n_outputs = 6

model = tf.keras.Sequential()
#model.add(layers.Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(layers.Conv2D(filters=32, kernel_size=(20, 1), input_shape=(n_timesteps,n_features, 1)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2D(filters=64, kernel_size=(10, 1))) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2D(filters=32, kernel_size=(5, 1))) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling2D(pool_size=(100, 1))) # Reducing dimensionality on time dimension
#model.add(layers.GlobalAveragePooling1D(data_format=None))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[39]:


model.summary()


# In[Output file]:

# Save best model and include early stopping
output_filename = 'CNN_EEG_classifier_avg.hdf5'
output_file = os.path.join(PATH_CODE, 'models_trained' , output_filename)


# In[Load model]:

if os.path.isfile(output_file):
    try:
        model.load_weights(output_file)
        print(f"Loaded weights from {output_file}")
    except Exception as e:
        print(repr(e))
        


# In[Train]:

do_train = True
if do_train:
    checkpointer = ModelCheckpoint(filepath = output_file, 
                                   monitor='val_accuracy', 
                                   verbose=1, 
                                   save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
    
    model.fit(x=train_generator, 
                       validation_data=val_generator,
                       epochs=50,
                       callbacks = [
                           checkpointer, 
                            earlystopper,
                       ]
                      )


# In[Compile]:

#Adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)
#model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[GradCam]

from grad_cam import grad_cam

input_model = model
input_image = X[0]
n_class = 0
layer_name = "conv1d_2"

gradcam = grad_cam(input_model, input_image, n_class, layer_name)


# In[Visualize Grad_Cam]
from matplotlib import pyplot as plt

plt.figure(figsize=(15, 10))
plt.title('GradCAM')
plt.axis('off')
#plt.imshow(np.transpose(X[0]))
plt.imshow(np.transpose(gradcam), cmap='jet', alpha=0.5)


# In[81]:


model.get_weights()


