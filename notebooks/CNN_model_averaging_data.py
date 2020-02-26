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


# In[33]:


train_generator = DataGenerator(list_IDs = IDs_train,
                                 main_labels = main_labels,
                                 label_dict = label_dict,
                                 binarizer_dict = binarizer_dict,
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
                                 n_classes=1, 
                                 shuffle=True)

val_generator = DataGenerator(list_IDs = IDs_val,
                                 main_labels = main_labels,
                                 label_dict = label_dict,
                                 binarizer_dict = binarizer_dict,
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
                                 n_classes=1, 
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
from tensorflow.keras import optimizers


# In[40]:


# Simple CNN model
n_timesteps = 501
n_features = 30
n_outputs = 6

model = tf.keras.Sequential()
model.add(layers.Conv1D(filters=48, kernel_size=20, input_shape=(n_timesteps,n_features)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling1D(pool_size=2))

model.add(layers.Conv1D(filters=64, kernel_size=10)) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling1D(pool_size=2))

model.add(layers.Conv1D(filters=96, kernel_size=5)) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling1D(pool_size=2))

#model.add(layers.Conv1D(filters=96, kernel_size=3, activation='relu'))
#model.add(layers.AveragePooling1D(pool_size=2))
#model.add(layers.MaxPooling1D(pool_size=2))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(60, activation='relu'))
model.add(layers.Dense(n_outputs, activation='softmax'))



# In[39]:


model.summary()


# In[Load model]:


# Save best model and include early stopping
output_filename = 'CNN_EEG_classifier_avg.hdf5'
output_file = os.path.join(PATH_CODE, 'models_trained' , output_filename)

if os.path.isfile(output_file):
    try:
        model.load_weights(output_file)
        print(f"Loaded weights from {output_file}")
    except Exception as e:
        print(repr(e))
        


# In[Compile]:

#Adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)
#model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[Train]:

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


# In[81]:


model.get_weights()


# In[ ]:





# # 1. Make clearer disctinction between training, test, validation data!
# To be sure the network is able to make predictions using unseen data, the dataset could be split according to persons! Unfortunately we only have data here for 57 persons (24 in group 1 and 33 in group 2).  
# This makes this approach a bit complicated
# 
# ## Idea for next steps:
# We could also loop through different test-person/train-person splits and independently train models on those datasets. Then, in the the end the outcome would be averaged over all of those. This way we would make better use of the data we have!

# In[11]:


groups = []
for ID in range(len(metadata)):
    if ID == 0:
        low = 0
    else:
        low = int(metadata[ID-1][2])
        
    groups.append(int(label_collection[low]/3))


# In[12]:


group_1_ids = np.where(np.array(groups) == 1)[0]
group_2_ids = np.where(np.array(groups) == 1)[0]


# In[13]:


# inspect group 2 ID's
group_2_ids


# In[15]:


n_epochs = 30 # Average over n_epochs epochs
n_class3_per_patient = 20
n_class13_per_patient = 10
n_class66_per_patient = 10

# Make selection 
group_1_ids = np.where(np.array(groups) == 1)[0]
group_2_ids = np.where(np.array(groups) == 2)[0]
keep_group1_for_test = 10
keep_group2_for_test = 10

# Initialize random numbers to get reproducible results 
np.random.seed(1)

# Make random selection
selected_ids_test = [x for x in np.concatenate([np.random.choice(group_1_ids, keep_group1_for_test),
                                   np.random.choice(group_2_ids, keep_group2_for_test)])]

selected_ids = [x for x in range(len(metadata)) if not x in selected_ids_test]


# In[111]:


selected_ids_test


# In[12]:


X_data, y_data = create_averaged_data(signal_collection, 
                         label_collection, 
                         metadata, 
                         n_epochs, 
                         selected_ids)


# In[18]:


X_test, y_test = create_averaged_data(signal_collection, 
                         label_collection, 
                         metadata, 
                         n_epochs, 
                         selected_ids_test)


# In[19]:


print(X_data.shape, len(y_data))
print(X_test.shape, len(y_test))


# In[32]:


# Check if samples of all label classes are present in both test and train
for i in [3, 6, 13, 26, 66, 132]:
    print(i, np.sum(np.array(y_test) == i))


# ## Normalization of EEG signals

# In[33]:


Xmean = np.concatenate([X_data, X_test]).mean()
X_data = X_data - Xmean
X_test = X_test - Xmean

Xmax = np.concatenate([X_data, X_test]).max()

X_data = X_data / Xmax
X_test = X_test / Xmax


# In[34]:


X_data.mean(), X_data.max(),X_data.min()


# ## Split training data
# ### Now the test dataset is already entirely seperate from the rest! 
# + validation data, used to monitor the model progress and avoid overfitting.
# + testing data, meant for final check on model performance.
# + --> Create validation and test data set from seperated data!!

# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
X_train, y_train = shuffle(X_data, y_data, random_state=1)


# In[36]:


print('Train set size:', X_train.shape[0])
print('Validation set size:', X_val.shape[0])
print('Test set size:', X_test.shape[0])
print()
print("X_train mean, min, max: ", np.mean(X_train), np.min(X_train), np.max(X_train))


# In[61]:


# Check if samples of all label classes are present in both test and train
for i in [3, 6, 13, 26, 66, 132]:
    print(i, np.sum(np.array(y_test) == i))

print()
for i in [3, 6, 13, 26, 66, 132]:
    print(i, np.sum(np.array(y_val) == i))


# ## Switch to 1-hot encoding for labels
# We have six categories or classes. Those are best represented by a so called **1-hot encoding**. This means nothing else than simply a binary 0-or-1 for every class.  
# The categories we have are:  
# 3 -> group 0 + stimuli "3"  
# 6 -> group 1 + stimuli "3"  
# 13 -> group 0 + stimuli "13"  
# 26 -> group 1 + stimuli "13"  
# 66 -> group 0 + stimuli "66"  
# 128 -> group 1 + stimuli "66"  
# 

# In[37]:


from sklearn.preprocessing import LabelBinarizer
label_transform = LabelBinarizer()

y_train_binary = label_transform.fit_transform(np.array(y_train).astype(int))
y_val_binary = label_transform.fit_transform(np.array(y_val).astype(int))
y_test_binary = label_transform.fit_transform(np.array(y_test).astype(int))


# In[38]:


y_val_binary[:10,:]


# In[39]:


# Show found labels:
label_transform.classes_


# Check distribution accross the 6 label categories:

# In[40]:


labels = list(label_transform.classes_)
frequencies = y_train_binary.mean(axis=0)
frequencies_df = pd.DataFrame(frequencies, index=labels, columns=['frequency'])
frequencies_df


# In[62]:


labels = list(label_transform.classes_)
frequencies = y_test_binary.mean(axis=0)
frequencies_df = pd.DataFrame(frequencies, index=labels, columns=['frequency'])
frequencies_df


# ### Note:
# We have more data on group 2 than on group 1. And far more data for stimuli 3 than for stimuli 13 and 66 (not surprising). 
# 
# --> post on balancing datasets: https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
# 
# ### Needs some thinking on how to balance the data set !
# e.g. by frequency dependend selection rule, or by defining a suitied special loss function....

# In[41]:


from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced'
                                               ,np.unique(y_train)
                                               ,y_train)


# In[42]:


class_weight


# In[43]:


class_weight = {0: class_weight[0],
               1: class_weight[1],
               2: class_weight[2],
               3: class_weight[3],
               4: class_weight[4],
               5: class_weight[5]}


# ## Define model architecture

# In[44]:


import tensorflow as tf
from tensorflow.keras import layers


# In[45]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
output_file = 'CNN_EEG_classifier_avg_02'

checkpointer = ModelCheckpoint(filepath = PATH_MODELS + output_file + ".hdf5", monitor='val_acc', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_acc', patience=5, verbose=1)


# In[46]:


# Simple CNN model
n_timesteps = 501
n_features = 30
n_outputs = 6

model = tf.keras.Sequential()
#model.add(layers.Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(layers.Conv1D(filters=32, kernel_size=20, input_shape=(n_timesteps,n_features)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling1D(pool_size=2))

model.add(layers.Conv1D(filters=64, kernel_size=10)) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling1D(pool_size=2))

model.add(layers.Conv1D(filters=64, kernel_size=5)) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling1D(pool_size=2))

#model.add(layers.Conv1D(filters=96, kernel_size=3, activation='relu'))
#model.add(layers.AveragePooling1D(pool_size=2))
#model.add(layers.MaxPooling1D(pool_size=2))

model.add(layers.Flatten())
model.add(layers.Dense(80, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_outputs, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[47]:


model.summary()


# In[48]:


epochs = 50
batch_size = 32

# fit network
model.fit(np.swapaxes(X_train,1,2), 
          y_train_binary, 
          validation_data=(np.swapaxes(X_val,1,2), y_val_binary), 
          epochs=epochs, 
          batch_size=batch_size,
          class_weight = class_weight,
          callbacks = [checkpointer, earlystopper])


# ## Seems to overfit on the training data and not be able to predict well the validation data

# In[49]:


# Evaluate the model
_, train_acc = model.evaluate(np.swapaxes(X_train,1,2), y_train_binary, verbose=0)
_, test_acc = model.evaluate(np.swapaxes(X_test,1,2), y_test_binary, verbose=0)


# In[50]:


print("Accuracy on train dataset:", train_acc)
print("Accuracy on test dataset:", test_acc)


# In[51]:


Xtest = np.swapaxes(X_test,1,2)

# Check model predictions:
y_pred_proba = model.predict_proba(Xtest)
y_pred_classes = model.predict_classes(Xtest)


# In[67]:


y_test_05 = np.array(y_test.copy())
y_test_05[y_test_05 == 3] = 0
y_test_05[y_test_05 == 6] = 1
y_test_05[y_test_05 == 13] = 2
y_test_05[y_test_05 == 26] = 3
y_test_05[y_test_05 == 66] = 4
y_test_05[y_test_05 == 132] = 5

print(y_test_05[:30].astype(int))
print(y_pred_classes[:30])


# In[68]:


# Calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test_05, y_pred_classes))


# ## Check if groups are predicted correctly:

# In[54]:


y_test_12 = np.array(y_test.copy())
y_test_12[y_test_12 == 132] = 2
y_test_12[y_test_12 == 66] = 1
y_test_12[y_test_12 == 26] = 2
y_test_12[y_test_12 == 13] = 1
y_test_12[y_test_12 == 6] = 2
y_test_12[y_test_12 == 3] = 1

y_pred_12 = y_pred_classes.copy()
y_pred_12[(y_pred_12 == 4) | (y_pred_12 == 2) | (y_pred_12 == 0)] = 6
y_pred_12[(y_pred_12 == 5) | (y_pred_12 == 3) | (y_pred_12 == 1)] = 2
y_pred_12[y_pred_12 == 6] = 1

print(y_test_12[:30].astype(int))
print(y_pred_12[:30])


# In[55]:


np.sum(y_test_12 == y_pred_12)/ y_pred_12.shape[0]


# ## Observation:
# So far this is not working as a proper discriminator between group1 and group2!  
# The test dataset contains about equal amounts of data from both groups, and the model does do better than random guessing (50% hits).

# In[69]:


# Confusion matrix:
M_confusion = metrics.confusion_matrix(y_test_05, y_pred_classes)
M_confusion


# In[80]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.utils.multiclass import unique_labels
    
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


# In[84]:


# Plot normalized confusion matrix
labels = list(label_transform.classes_)

plot_confusion_matrix(y_test_05, y_pred_classes, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('confusion_matrix.pdf')


# ### Interestingly, it might be that the model at least does see a difference between the types of stimuli!

# In[71]:


y_test_bak_dak = np.array(y_test.copy())
y_test_bak_dak[y_test_bak_dak == 132] = 2
y_test_bak_dak[y_test_bak_dak == 66] = 2
y_test_bak_dak[y_test_bak_dak == 26] = 2
y_test_bak_dak[y_test_bak_dak == 13] = 2
y_test_bak_dak[y_test_bak_dak == 6] = 1
y_test_bak_dak[y_test_bak_dak == 3] = 1

y_pred_bak_dak = y_pred_classes.copy()
y_pred_bak_dak[y_pred_bak_dak == 2] = 2
y_pred_bak_dak[y_pred_bak_dak == 5] = 2
y_pred_bak_dak[y_pred_bak_dak == 4] = 2
y_pred_bak_dak[y_pred_bak_dak == 3] = 2
y_pred_bak_dak[y_pred_bak_dak == 1] = 1
y_pred_bak_dak[y_pred_bak_dak == 0] = 1

print(y_test_bak_dak[:30].astype(int))
print(y_pred_bak_dak[:30])


# In[72]:


np.sum(y_test_bak_dak == y_pred_bak_dak)/ y_pred_bak_dak.shape[0]


# ## Interestingly though,...
# That would mean that the model is correct in >95% of all cases in distinguishing 'bak' from 'dak'.  
# --> **Careful: Needs to be checked if my assumptions about the stimuli are correct...**

# # 2. Alternative model architecture
# + Compression in time axis only after 3 convolutional steps
# + Contains batchNormalization layers

# In[85]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
output_file = 'CNN_EEG_classifier_avg_03'

checkpointer = ModelCheckpoint(filepath = PATH_MODELS + output_file + ".hdf5", monitor='val_acc', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_acc', patience=5, verbose=1)


# In[86]:


# Simple CNN model
n_timesteps = 501
n_features = 30
n_outputs = 6

model = tf.keras.Sequential()
#model.add(layers.Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(layers.Conv1D(filters=32, kernel_size=20, input_shape=(n_timesteps,n_features)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv1D(filters=64, kernel_size=10)) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv1D(filters=32, kernel_size=5)) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling1D(pool_size=4))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[87]:


model.summary()


# In[88]:


epochs = 50
batch_size = 32

# fit network
model.fit(np.swapaxes(X_train,1,2), 
          y_train_binary, 
          validation_data=(np.swapaxes(X_val,1,2), y_val_binary), 
          epochs=epochs, 
          batch_size=batch_size,
          class_weight = class_weight,
          callbacks = [checkpointer, earlystopper])


# ## Seems to overfit on the training data and not be able to predict well the validation data

# In[89]:


# Evaluate the model
_, train_acc = model.evaluate(np.swapaxes(X_train,1,2), y_train_binary, verbose=0)
_, test_acc = model.evaluate(np.swapaxes(X_test,1,2), y_test_binary, verbose=0)


# In[90]:


print("Accuracy on train dataset:", train_acc)
print("Accuracy on test dataset:", test_acc)


# In[91]:


Xtest = np.swapaxes(X_test,1,2)

# Check model predictions:
y_pred_proba = model.predict_proba(Xtest)
y_pred_classes = model.predict_classes(Xtest)


# In[93]:


y_test_05 = np.array(y_test.copy())
y_test_05[y_test_05 == 3] = 0
y_test_05[y_test_05 == 6] = 1
y_test_05[y_test_05 == 13] = 2
y_test_05[y_test_05 == 26] = 3
y_test_05[y_test_05 == 66] = 4
y_test_05[y_test_05 == 132] = 5

print(y_test_05[:20].astype(int))
print(y_pred_classes[:20])


# In[94]:


# Calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test_05, y_pred_classes))


# ## Check if groups are predicted correctly:

# In[98]:


y_test_12 = np.array(y_test.copy())
y_test_12[y_test_12 == 132] = 2
y_test_12[y_test_12 == 66] = 1
y_test_12[y_test_12 == 26] = 2
y_test_12[y_test_12 == 13] = 1
y_test_12[y_test_12 == 6] = 2
y_test_12[y_test_12 == 3] = 1

y_pred_12 = y_pred_classes.copy()
y_pred_12[(y_pred_12 == 4) | (y_pred_12 == 2) | (y_pred_12 == 0)] = 6
y_pred_12[(y_pred_12 == 5) | (y_pred_12 == 3) | (y_pred_12 == 1)] = 2
y_pred_12[y_pred_12 == 6] = 1

print(y_test_12[:30].astype(int))
print(y_pred_12[:30])


# In[99]:


np.sum(y_test_12 == y_pred_12)/ y_pred_12.shape[0]


# In[109]:


# Plot normalized confusion matrix
labels = list(label_transform.classes_)

plot_confusion_matrix(y_test_12, y_pred_12, classes=['risk', 'non-risk'], normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('confusion_matrix12.pdf')


# ## Note:
# Groups are not correctly predicted. It's not better than trowing dices...

# ## Check if stimuli types are predicted correctly:

# In[102]:


y_test_123 = np.array(y_test.copy())
y_test_123[y_test_123 == 6] = 1
y_test_123[y_test_123 == 3] = 1
y_test_123[y_test_123 == 26] = 2
y_test_123[y_test_123 == 13] = 2
y_test_123[y_test_123 == 132] = 3
y_test_123[y_test_123 == 66] = 3

y_pred_123 = y_pred_classes.copy()
y_pred_123[(y_pred_123 == 1) | (y_pred_123 == 0)] = 1
y_pred_123[(y_pred_123 == 3) | (y_pred_123 == 2)] = 2
y_pred_123[(y_pred_123 == 5) | (y_pred_123 == 4)] = 3

print(y_test_123[:30].astype(int))
print(y_pred_123[:30])


# In[103]:


np.sum(y_test_123 == y_pred_123)/ y_pred_123.shape[0]


# In[110]:


# Plot normalized confusion matrix
labels = list(label_transform.classes_)

plot_confusion_matrix(y_test_123, y_pred_123, classes=['standard', 'deviant 1', 'deviant 2'], normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('confusion_matrix123.pdf')


# ## Observation:
# So far this is not working as a proper discriminator between group1 and group2!  
# The test dataset contains about equal amounts of data from both groups, and the model does do better than random guessing (50% hits).

# In[100]:


# Confusion matrix:
M_confusion = metrics.confusion_matrix(y_test_05, y_pred_classes)
M_confusion


# In[101]:


# Plot normalized confusion matrix
labels = list(label_transform.classes_)

plot_confusion_matrix(y_test_05, y_pred_classes, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('confusion_matrix2.pdf')


# # 3. Train on only stimuli labels (not including patient groups)

# In[143]:


old_label = [3,6,13,26,66,132]
new_label = [3,3,13,13,66,66]

y_train_123 = np.zeros((len(y_train)))
for i, old in enumerate(old_label):
    y_train_123[np.where(np.array(y_train) == old)[0]] = new_label[i]

y_val_123 = np.zeros((len(y_val)))
for i, old in enumerate(old_label):
    y_val_123[np.where(np.array(y_val) == old)[0]] = new_label[i]
    
y_test_123 = np.zeros((len(y_test)))
for i, old in enumerate(old_label):
    y_test_123[np.where(np.array(y_test) == old)[0]] = new_label[i]


# In[144]:


label_transform = LabelBinarizer()

y_train123_binary = label_transform.fit_transform(np.array(y_train_123).astype(int))
y_val123_binary = label_transform.fit_transform(np.array(y_val_123).astype(int))
y_test123_binary = label_transform.fit_transform(np.array(y_test_123).astype(int))


# In[117]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
output_file = 'CNN_EEG_classifier_avg_123'

checkpointer = ModelCheckpoint(filepath = PATH_MODELS + output_file + ".hdf5", monitor='val_acc', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_acc', patience=5, verbose=1)


# In[118]:


# Simple CNN model
n_timesteps = 501
n_features = 30
n_outputs = 3

model = tf.keras.Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=20, input_shape=(n_timesteps,n_features)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv1D(filters=64, kernel_size=10)) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv1D(filters=32, kernel_size=5)) #, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.AveragePooling1D(pool_size=4))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[119]:


model.summary()


# In[146]:


y_train123_binary[:10,:]


# In[ ]:





# In[135]:


epochs = 50
batch_size = 32

# Fit network
model.fit(np.swapaxes(X_train,1,2), 
          y_train123_binary, 
          validation_data=(np.swapaxes(X_val,1,2), y_val123_binary), 
          epochs=epochs, 
          batch_size=batch_size,
          class_weight = class_weight,
          callbacks = [checkpointer, earlystopper])


# In[151]:


# Check model predictions:
y_pred_123 = model.predict_classes(Xtest)


# In[152]:


y_pred_123 = y_pred_123.copy()
y_pred_123[y_pred_123 == 2] = 66
y_pred_123[y_pred_123 == 1] = 13
y_pred_123[y_pred_123 == 0] = 3


# In[150]:


y_test_123[:20]


# In[153]:


# Plot normalized confusion matrix
plot_confusion_matrix(y_test_123, y_pred_123, classes=['standard', 'deviant 1', 'deviant 2'], normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('confusion_matrix123_training123.pdf')

