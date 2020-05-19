#!/usr/bin/env python
# -*- coding: utf-8 -*-

# In[Import packages]:
import os
import sys
import numpy as np
import pandas as pd

# In[Import local packages]
cwd = os.getcwd()
# sys.path.insert(0, os.path.dirname(os.getcwd()))

# Using downloaded mcfly repository to be able to modify the source code
mcfly_path = os.path.abspath(os.path.join(cwd, "..", "tools", "mcfly"))
if not mcfly_path in sys.path:
    sys.path.append(mcfly_path)
import mcfly


# In[2]:


from config import PATH_CODE, PATH_DATA

do_load = True
do_train = False

# ## Load pre-processed dataset
# + See notebook for preprocessing: ePODIUM_prepare_data_for_ML.ipynb.ipynb


# ts_type = "test"
ts_type = "benchmark1"
# n_samples = 200
n_samples = 1000
ignore_noise = False
# ignore_noise_network = True

PATH_PLOTS = "plots"

PATH_DATA_processed = os.path.join(PATH_DATA, "test_data")


# In[Load Files]:

x_data = np.load(os.path.join(
    PATH_DATA_processed,
    f"x_data_{ts_type}_s{n_samples}_n{int(not ignore_noise)}.npy"))
y_data = np.load(os.path.join(
    PATH_DATA_processed,
    f"y_data_{ts_type}_s{n_samples}_n{int(not ignore_noise)}.npy"))


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


# In[Randomize ids]:

# ids_train = np.array(ids_train)[np.random.permutation(len(ids_train))]
# ids_val = np.array(ids_train)[np.random.permutation(len(ids_val))]
# ids_test = np.array(ids_train)[np.random.permutation(len(ids_test))]

np.random.shuffle(ids_train)
np.random.shuffle(ids_val)
np.random.shuffle(ids_test)


# In[Binarize labels]

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(label_collection)
print(lb.classes_)

binary_y_data = lb.fit_transform(y_data)



# In[Split datasets]

# NOTE: mcfly takes data in the shape of length timeseries and channels

def exchange_channels(data):
    return data.reshape(
        (data.shape[0], data.shape[2] * data.shape[1])).reshape((
            data.shape[0], data.shape[2], data.shape[1]), order = 'F')

def check_reshape(o_data, rs_data):
    for i in range(o_data.shape[0]):
        for j in range(o_data.shape[1]):
            for k in range(o_data.shape[2]):
                if o_data[i][j][k] != rs_data[i][k][j]:
                    return False
    return True

def prepare_sets(indices, full_x_data, full_y_data):
    x_set = list()
    y_set = list()
    for idx in indices:
        x_set.append(full_x_data[idx])
        y_set.append(full_y_data[idx])
    x_set = np.array(x_set)
    # x_set = x_set.reshape(np.concatenate((x_set.shape, [1])))
    y_set = np.array(y_set)
    return x_set, y_set

print(x_data.shape)
x_data = exchange_channels(x_data)
print(x_data.shape)

x_set_train, y_set_train = prepare_sets(ids_train, x_data, binary_y_data)
x_set_val, y_set_val = prepare_sets(ids_val, x_data, binary_y_data)
x_set_test, y_set_test = prepare_sets(ids_test, x_data, binary_y_data)


# In[Initialize mcfly]:

from datetime import datetime

data_type = f"{ts_type}-noise{int(not ignore_noise)}"

# test_type = "long"
# test_type = "short"
test_type = "feature_test"

if test_type == "long":
    NR_MODELS = 30
    NR_EPOCHS = 30
    EARLY_STOPPING = 10
    SUBSET_SIZE = 600
    MODEL_TYPES = ["CNN", "InceptionTime", "DeepConvLSTM", "ResNet", "FCN", "Encoder"]
elif test_type == "short":
    NR_MODELS = 30
    NR_EPOCHS = 20
    EARLY_STOPPING = 5
    SUBSET_SIZE = 300
    MODEL_TYPES = ["CNN", "InceptionTime", "DeepConvLSTM", "ResNet", "FCN", "Encoder"]
elif test_type == "feature_test":
    NR_MODELS = 1
    NR_EPOCHS = 20
    EARLY_STOPPING = 5
    SUBSET_SIZE = 300
    # MODEL_TYPES = ["CNN", "InceptionTime", "DeepConvLSTM", "ResNet", "FCN", "Encoder"]
    MODEL_TYPES = ["CNN_2D"]

train_type = "models{}-epochs{}-e_stop{}-subset{}".format(
    NR_MODELS,
    NR_EPOCHS,
    EARLY_STOPPING,
    SUBSET_SIZE)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

outputfile_name = f"{timestamp}-{data_type}-{train_type}-All-FCN-Encoder"
# outputfile_name = f"{timestamp}-{data_type}-{train_type}-Encoder.json"

num_classes = binary_y_data.shape[1]
metric = 'accuracy'


# In[Generate models]:
models = mcfly.modelgen.generate_models(
    x_set_train.shape,
    number_of_classes=num_classes,
    number_of_models = NR_MODELS,
    model_types = MODEL_TYPES,
    encoder_max_layers = 9,
    metrics=[metric])


# In[Train models]:

# Define directory where the results, e.g. json file, will be stored
resultpath = os.path.join('.', 'models')
if not os.path.exists(resultpath):
    os.makedirs(resultpath)


from mcfly.find_architecture import train_models_on_samples

outputfile = os.path.join(resultpath, f"{outputfile_name}.json")
histories, val_accuracies, val_losses = train_models_on_samples(
    x_set_train, y_set_train,
    x_set_val, y_set_val,
    models, nr_epochs=NR_EPOCHS,
    subset_size=SUBSET_SIZE,
    early_stopping_patience = EARLY_STOPPING,
    verbose=True,
    outputfile=outputfile,
    metric=metric)

print('Details of the training process were stored in ',outputfile)


# In[Print and store table]

metric = 'accuracy'
modelcomparisons = pd.DataFrame({
    'model':[str(params) for model, params, model_types in models],
    'model-type':[str(model_types) for model, params, model_types in models],
    'train_{}'.format(metric): [history.history[metric][-1] for history in histories],
    'train_loss': [history.history['loss'][-1] for history in histories],
    'val_{}'.format(metric): [history.history['val_{}'.format(metric)][-1] for history in histories],
    'val_loss': [history.history['val_loss'][-1] for history in histories]
    })
modelcomparisons.to_csv(os.path.join(resultpath, f"{outputfile_name}.csv"))

print(modelcomparisons)





