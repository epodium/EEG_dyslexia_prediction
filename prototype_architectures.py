#!/usr/bin/env python
# -*- coding: utf-8 -*-

# In[Import packages]:
import os
import sys
import numpy as np
import pandas as pd

from real_data_utils import prepare_set, exchange_channels, prepare_generators
import benchmark_data_utils


# In[Import local packages]
cwd = os.getcwd()
# sys.path.insert(0, os.path.dirname(os.getcwd()))

# Using downloaded mcfly repository to be able to modify the source code
mcfly_path = os.path.abspath(os.path.join(cwd, "..", "tools", "mcfly"))
if not mcfly_path in sys.path:
    sys.path.append(mcfly_path)
import mcfly


# In[2]:


from config import PATH_CODE, PATH_DATA, ROOT


real_data = True
# real_data = False

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


    # In[Load Files]:

if real_data:
    import fnmatch

    PATH_DATA_processed = os.path.join(PATH_DATA, dataset_folder)

    dirs = os.listdir(PATH_DATA_processed)
    files_npy = fnmatch.filter(dirs, "*.npy")
    files_csv = fnmatch.filter(dirs, "*.csv")

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

    label_dict  = {
        '66dys0_risk0': '0',
        '66dys0_risk1': '0',
        '66dys1_risk0': '1',
        '66dys1_risk1': '1',
    }

    ignore_labels = [
        '3dys0_risk0', '3dys0_risk1', '3dys1_risk0', '3dys1_risk1',
        '13dys0_risk0', '13dys0_risk1', '13dys1_risk0', '13dys1_risk1']

    np.random.seed(1098)
    # split_ratio = (0.7, 0.15, 0.15)
    split_ratio = (0.8, 0.2, 0)
    train_generator, val_generator, test_generator = prepare_generators(
        PATH_DATA_processed,
        main_label_dict,
        label_dict,
        split_ratio,
        ignore_labels)

    # x_set_train, y_set_train = unroll_generator(train_generator)
    # x_set_val, y_set_val = unroll_generator(val_generator)
    x_set_train = train_generator
    y_set_train = None

    # XXX: It should generally work, but this is not really a safe assumption
    num_classes = len(train_generator.binarizer_dict)

    # XXX: McFly shouldn't need the first item
    input_shape = (
        len(train_generator.list_IDs),
        train_generator.n_timepoints,
        train_generator.n_channels)

    x_set_val = val_generator
    y_set_val = None

    data_type = f"{dataset_folder}"




else:
    x_data, y_data, data_type = benchmark_data_utils.load_data(
        PATH_DATA_processed,
        ts_type,
        n_samples,
        ignore_noise)

    # In[Separate data by labels]
    label_collection, label_ids_dict = benchmark_data_utils.collect_labels(y_data)


    # In[Split labels]:

    ids_train, ids_val, ids_test = benchmark_data_utils.split_labels(label_collection, label_ids_dict, 1098)


    # In[Randomize ids]:

    np.random.shuffle(ids_train)
    np.random.shuffle(ids_val)
    np.random.shuffle(ids_test)


    # In[Binarize labels]

    binary_y_data = benchmark_data_utils.binarize_labels(y_data, label_collection)

    # In[Split datasets]

    # NOTE: mcfly takes data in the shape of length timeseries and channels

    print(x_data.shape)
    x_data = exchange_channels(x_data)
    print(x_data.shape)

    x_set_train, y_set_train = prepare_set(ids_train, x_data, binary_y_data)
    x_set_val, y_set_val = prepare_set(ids_val, x_data, binary_y_data)
    x_set_test, y_set_test = prepare_set(ids_test, x_data, binary_y_data)

    input_shape = x_set_train.shape
    num_classes = y_set_train.shape[1]


# In[Initialize mcfly]:

from datetime import datetime

test_type = "long"
# test_type = "short"
test_type = "feature_test"

if test_type == "long":
    NR_MODELS = 30
    NR_EPOCHS = 30
    EARLY_STOPPING = 10
    SUBSET_SIZE = 600
    MODEL_TYPES = ["CNN", "InceptionTime", "DeepConvLSTM", "ResNet", "FCN", "Encoder"]
    model_types = "All-FCN-Encoder"
elif test_type == "short":
    NR_MODELS = 18
    NR_EPOCHS = 20
    EARLY_STOPPING = 5
    SUBSET_SIZE = 300
    MODEL_TYPES = ["CNN", "InceptionTime", "DeepConvLSTM", "ResNet", "FCN", "Encoder"]
    model_types = "All-FCN-Encoder"
elif test_type == "feature_test":
    NR_MODELS = 24
    NR_EPOCHS = 30
    EARLY_STOPPING = 10
    SUBSET_SIZE = 600
    # MODEL_TYPES = ["CNN", "InceptionTime", "Encoder", "Encoder_2D"]
    # model_types = "cnn-inception-encoders"
    # MODEL_TYPES = ["CNN_2D"]
    MODEL_TYPES = ["Encoder", "Encoder_2D"]
    model_types = "encoders"
    # MODEL_TYPES = ["Encoder_2D"]
    # model_types = "encoder2d"

train_type = "models{}-epochs{}-e_stop{}-subset{}".format(
    NR_MODELS,
    NR_EPOCHS,
    EARLY_STOPPING,
    SUBSET_SIZE)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

outputfile_name = f"{timestamp}-{data_type}-{train_type}-{model_types}"
# outputfile_name = f"{timestamp}-{data_type}-{train_type}-Encoder.json"

metric = 'accuracy'




# In[Generate models]:
models = mcfly.modelgen.generate_models(
    input_shape,
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






