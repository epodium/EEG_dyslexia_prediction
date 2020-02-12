#!/usr/bin/env python
# coding: utf-8

# # Prepare EEG data for training of machine-learning models
# + Import data.
# + Apply filters (bandpass).
# + Detect potential bad channels and replace them by interpolation.
# + Detect potential bad epochs and remove them.

# ## Import packages & links

# In[1]:


# Import packages
import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.getcwd()))

import mne
#%matplotlib inline
#from mayavi import mlab


# In[10]:


from config import ROOT, PATH_CODE, PATH_DATA, PATH_OUTPUT, PATH_METADATA


# ### Update path!

# In[37]:


PATH_CNTS = os.path.join(PATH_DATA, "41mnd mmn")
PATH_OUTPUT = os.path.join(PATH_DATA, 'processed_data_41mnd')


# In[11]:


filename_labels = PATH_METADATA + "Screening_children5a_summary_new.txt" 
metadata = pd.read_csv(filename_labels, sep='\t')
metadata.head()


# In[12]:


metadata.shape


# ## Search all *.cnt files and check for how many we have a label

# In[13]:


import fnmatch
import warnings
warnings.filterwarnings('ignore')

import helper_functions

dirs = os.listdir(PATH_CNTS)
cnt_files = fnmatch.filter(dirs, "*.cnt")


# In[14]:


found_ids = [x[:3] for x in cnt_files]
idx = np.where(np.array(found_ids) == '036')[0]
[cnt_files[x] for x in idx]


# In[15]:


metadata[metadata['id_child'] == '036']['groupDDP'].values[0]


# In[16]:


labels = []

found_ids = [x[:3] for x in cnt_files]
for ID in list(set(found_ids)): 
    idx = np.where(np.array(found_ids) == ID)[0]
    filenames = [cnt_files[x] for x in idx]
    label = metadata[metadata['id_child'] == ID]['groupDDP'].values[0]
    label_risk = metadata[metadata['id_child'] == ID]['assignment4'].values[0]
    if label == '1FRdys':
        label = 1
    elif label == '2FRndys':
        label = 0
    elif label == '3Ctrl': #TODO: check if this is correct!
        label = 0
    labels.append([ID, label, label_risk, filenames])


# In[8]:


labels[:10]


# In[17]:


len(labels), len(list(set(found_ids)))


# ### Count number (and type) of labels found:

# In[18]:


labels_known = 0
labels_unknown = 0
labels_type = []

for x in labels:
    if x[1] == 1: #'dyslexic'
        labels_known += 1
        labels_type.append(1)
    elif x[1] == 0: #'non-dyslexic'
        labels_known += 1
        labels_type.append(0)
    else: # missing or unclear
        labels_unknown += 1  
        labels_type.append('missing')
        
print("Data with proper labels:", labels_known, "||| Data without proper label:", labels_unknown)     


# In[19]:


print("Data for 'dyslexic':", labels_type.count(1))
print("Data for 'non-dyslexic':", labels_type.count(0))


# In[20]:


# Check types of risk group labels found
labels_risktype = [x[2] for x in labels]
list(set(labels_risktype))


# In[21]:


metadata['atRiskOrNotDDP'][:10]


# In[22]:


group_notrisk = np.array(1*((metadata['atRiskOrNotDDP'] == 'notAtRisk')
                   | (metadata['assignment4'].isin(['notAtRisk_rest', 'notAtRisk_highestScores']))))

group_risk = np.array(1*((metadata['atRiskOrNotDDP'] == 'atRisk')
                   | (metadata['assignment4'] == 'at risk')))


# In[23]:


np.sum(group_risk) + np.sum(group_notrisk)


# In[24]:


label_risk = group_notrisk + 2*group_risk 
label_risk[label_risk == 3] = 2
label_risk = label_risk -1


# In[25]:


label_risk


# In[26]:


group_notdys = np.array(1*(metadata['groupDDP'].isin(['1FRdys', '3Ctrl'])))

group_dys = np.array(1*(metadata['groupDDP'] == '2FRndys'))


# In[27]:


np.sum(group_notdys) + np.sum(group_dys)


# In[28]:


label_dys = group_notdys + 2*group_dys 
label_dys[label_dys == 3] = 2
label_dys = label_dys -1


# In[29]:


label_dys


# ## create Dataframe with labels to be used

# In[30]:


labels_final = pd.DataFrame(data=metadata['id_child'].values, columns=['id_child'])
labels_final['label_dys'] = label_dys
labels_final['label_risk'] = label_risk
labels_final.head()


# In[31]:


print("Data for 'at risk':", labels_risktype.count('atRisk'))
print("Data for 'notAtRisk_rest':", labels_risktype.count('notAtRisk_rest'))
print("Data for 'notAtRisk_highestScores':", labels_risktype.count('notAtRisk_highestScores'))


# In[32]:


labels_risktype = [x[2] for x in labels if x[1] in [1,0]]
print("Data for 'at risk':", labels_risktype.count('atRisk'))
print("Data for 'notAtRisk_rest':", labels_risktype.count('notAtRisk_rest'))
print("Data for 'notAtRisk_highestScores':", labels_risktype.count('notAtRisk_highestScores'))


# In[33]:


metadata.loc[(metadata['groupDDP'].isin(['1FRdys', '2FRndys', '3Ctrl']) 
              & metadata['assignment4'].isin(['at risk', 'notAtRisk_rest', 'notAtRisk_highestScores', ]))]


# ## Custom cnt-file import function:

# In[34]:


def read_cnt_file(file,
                  label_group,
                  event_idx = [3, 13, 66],
                  channel_set = "30",
                  tmin = -0.2,
                  tmax = 0.8,
                  lpass = 0.5, 
                  hpass = 40, 
                  threshold = 5, 
                  max_bad_fraction = 0.2,
                  max_bad_channels = 2):
    """ Function to read cnt file. Run bandpass filter. 
    Then detect and correct/remove bad channels and bad epochs.
    Store resulting epochs as arrays.
    
    Args:
    --------
    file: str
        Name of file to import.
    label_group: int
        Unique ID of specific group (must be >0).
    channel_set: str
        Select among pre-defined channel sets. Here: "30" or "62"
    """
    
    if channel_set == "30":
        channel_set = ['O2', 'O1', 'OZ', 'PZ', 'P4', 'CP4', 'P8', 'C4', 'TP8', 'T8', 'P7', 
                       'P3', 'CP3', 'CPZ', 'CZ', 'FC4', 'FT8', 'TP7', 'C3', 'FCZ', 'FZ', 
                       'F4', 'F8', 'T7', 'FT7', 'FC3', 'F3', 'FP2', 'F7', 'FP1']
    elif channel_set == "62":
        channel_set = ['O2', 'O1', 'OZ', 'PZ', 'P4', 'CP4', 'P8', 'C4', 'TP8', 'T8', 'P7', 
                       'P3', 'CP3', 'CPZ', 'CZ', 'FC4', 'FT8', 'TP7', 'C3', 'FCZ', 'FZ', 
                       'F4', 'F8', 'T7', 'FT7', 'FC3', 'F3', 'FP2', 'F7', 'FP1', 'AFZ', 'PO3', 
                       'P1', 'POZ', 'P2', 'PO4', 'CP2', 'P6', 'M1', 'CP6', 'C6', 'PO8', 'PO7', 
                       'P5', 'CP5', 'CP1', 'C1', 'C2', 'FC2', 'FC6', 'C5', 'FC1', 'F2', 'F6', 
                       'FC5', 'F1', 'AF4', 'AF8', 'F5', 'AF7', 'AF3', 'FPZ']
    else:
        print("Predefined channel set given by 'channel_set' not known...")
        
    
    # Initialize array
    signal_collection = np.zeros((0,len(channel_set),501))
    label_collection = [] #np.zeros((0))
    
    # Import file
    try:
        data_raw = mne.io.read_raw_cnt(file, eog='auto', preload=True)
    except ValueError:
        print("ValueError")
        print("Could not load file:", file)
        return None, None
    
    # Band-pass filter (between 0.5 and 40 Hz. was 0.5 to 30Hz in Stober 2016)
    data_raw.filter(0.5, 40, fir_design='firwin')

    events_from_annot, event_dict = mne.events_from_annotations(data_raw)
    
    # TODO: check here already if event_idx's in event_dict
    
    # Set baseline:
    baseline = (None, 0)  # means from the first instant to t = 0

    # Select channels to exclude (if any)
    channels_exclude = [x for x in data_raw.ch_names if x not in channel_set]
    channels_exclude = [x for x in channels_exclude if x not in ['HEOG', 'VEOG']]#, 'STI 014']]
    
    for event_id in event_idx:
        if str(event_id) in event_dict:
            # Pick EEG channels 
            picks = mne.pick_types(data_raw.info, meg=False, eeg=True, stim=False, eog=False,
                               #exclude=data_exclude)#'bads'])
                                   include=channel_set, exclude=channels_exclude)#'bads'])

            epochs = mne.Epochs(data_raw, events=events_from_annot, event_id=event_dict, 
                                tmin=tmin, tmax=tmax, proj=True, picks=picks,
                                baseline=baseline, preload=True, event_repeated='merge', verbose=False)

            # Detect potential bad channels and epochs
            bad_channels, bad_epochs = helper_functions.select_bad_epochs(epochs, 
                                                                          event_id, 
                                                                          threshold = threshold, 
                                                                          max_bad_fraction = max_bad_fraction)

            # Interpolate bad channels
            # ------------------------------------------------------------------
            if len(bad_channels) > 0: 
                if len(bad_channels) > max_bad_channels: 
                    print(20*'--')
                    print("Found too many bad channels (" + str(len(bad_channels)) + ")")
                    return None, None
                else:
                    # Mark bad channels:
                    data_raw.info['bads'] = bad_channels
                    # Pick EEG channels:
                    picks = mne.pick_types(data_raw.info, meg=False, eeg=True, stim=False, eog=False,
                                       #exclude=data_exclude)#'bads'])
                                       include=channel_set, exclude=channels_exclude)#'bads'])
                    epochs = mne.Epochs(data_raw, events=events_from_annot, event_id=event_dict, 
                                        tmin=tmin, tmax=tmax, proj=True, picks=picks,
                                        baseline=baseline, preload=True, verbose=False)
                    # Interpolate bad channels using functionality of 'mne'
                    epochs.interpolate_bads()

            # Get signals as array and add to total collection
            signals_cleaned = epochs[str(event_id)].drop(bad_epochs).get_data()
            signal_collection = np.concatenate((signal_collection, signals_cleaned), axis=0)
            #label_collection = np.concatenate((label_collection, event_id*label_group*np.ones((signals_cleaned.shape[0]))), axis=0)
            label_collection += [str(event_id) + label_group] * signals_cleaned.shape[0]

    return signal_collection, label_collection#.astype(int)


# In[28]:


found_ids = [x[:3] for x in cnt_files]
idx = np.where(np.array(found_ids) == '036')[0]
[cnt_files[x] for x in idx]


# In[39]:


filename = os.path.join(PATH_DATA_01, '035_17_jc_mmn25.cnt')
data_raw = mne.io.read_raw_cnt(filename, eog=['HEOG', 'VEOG'], preload=True) #, montage=None, eog='auto', preload=True)
#data_raw.filter(0.5, 40, fir_design='firwin')


# In[139]:


data_raw.info #.get_data().shape


# In[140]:


events_from_annot, event_dict = mne.events_from_annotations(data_raw)
print(event_dict)
print(events_from_annot[:5])


# In[126]:


# channel names for 62 EEG channel case: 
print(epochs.ch_names)


# ## Check how many EEG channels the cnt-files feature... 

# In[142]:


format_collection = []
for i, filename in enumerate(cnt_files):
    # Import file 
    file = os.path.join(PATH_CNTS, filename)
    try:
        data_raw = mne.io.read_raw_cnt(file, eog='auto', preload=True)
        format_collection.append((i, len(data_raw.ch_names)))
        print(i, len(data_raw.ch_names))
    except ValueError:
        print("ValueError for file:", filename)
        format_collection.append((i, 0))


# In[144]:


a,b = zip(*format_collection)
len(np.where((np.array(b) == 64))[0]), len(np.where((np.array(b) == 32))[0]), len(a)


# So far we 'only' have about 60 cnt-files of which we have a label ("risk group" vs "no risc group").
# And only 42 of them feature 62 EEG channels. I hence switched to 30 EEG channels and picked the ones that are present in all patient datasets.

# # Workflow data processing
# 1. Load cnt files.
# 2. Select same number of channels (here: 30 same channels which exist for both 30 and 62 channel data)
# 3. Preprocess raw data (bandpass + detect outliers and 'bad' epochs).
# 4. Store epoch data and event type as array

# ## LABELS:
# + After Karin's search we have proper labels for much more files!  
# 

# In[38]:


def standardize_EEG(data_array,
                    std_aim = 1,
                   centering = 'per_channel',
                   scaling = 'global'):
    """ Center data around 0 and adjust standard deviation.
    
    Args:
    --------
    centering: str
        Specify if centering should be done "per_channel", or "global".
    scaling: str
        Specify if scaling should be done "per_channel", or "global".
    """
    if centering == 'global':
        data_mean = data_array.mean()

        # Center around 0
        data_array = data_array - data_mean

    elif centering == 'per_channel':       
        for i in range(data_array.shape[1]):
            
            data_mean = data_array[:,i,:].mean()

            # Center around 0
            data_array[:,i,:] = data_array[:,i,:] - data_mean

    else:
        print("Centering method not known.")
        return None
        
    if scaling == 'global':
        data_std = data_array.std()
        
        # Adjust std to std_aim
        data_array = data_array * std_aim/data_std
    
    elif scaling == 'per_channel':
        for i in range(data_array.shape[1]):
            
            data_std = data_array[:,i,:].std()

            # Adjust std to std_aim
            data_array[:,i,:] = data_array[:,i,:] * std_aim/data_std
    else:
        print("Given method is not known.")
        return None
    
    
    return data_array


# In[39]:


# Initialize array
signal_collection = np.zeros((0,30,501)) #62
label_collection = []
ID_collection = []
metadata_collection = []

collect_in_one_array = False

for i, filename in enumerate(cnt_files):
    
    # First check if we have proper label for that file
    # -----------------------------------------------------------
    
    ID = filename[:3]
    label = labels_final[labels_final['id_child'] == ID]['label_dys'].values[0]
    label_risk = labels_final[labels_final['id_child'] == ID]['label_risk'].values[0]
    #label = metadata[metadata['id_child'] == ID]['groupDDP'].values[0]
    #label_risk = metadata[metadata['id_child'] == ID]['assignment4'].values[0]
    
    if (label < 0) or (label_risk < 0):
        print("No proper label found for file: ", filename)
    else:
        #label_group = int(metadata[metadata["file"].str.match(filename[:-4])]['group'])
        label_group = 'dys' + str(label) + '_risk' + str(label_risk)
        
        print(40*"=")
        print("Importing file: ",filename)
        print("Data belongs into group: ", label_group)

        # Import data and events
        file = os.path.join(PATH_CNTS, filename)

        signal_collect, label_collect = read_cnt_file(file, 
                                                      label_group,
                                                      event_idx = [3, 13, 66],
                                                      channel_set = "30",
                                                      tmin = -0.2,
                                                      tmax = 0.8,
                                                      lpass = 0.5, 
                                                      hpass = 40, 
                                                      threshold = 5, 
                                                      max_bad_fraction = 0.2)
        
        
        # Standardize data
        # --------------------------------------------------------
        if signal_collect is not None:
            signal_collect = standardize_EEG(signal_collect,
                                 std_aim = 1,                   
                                 centering = 'per_channel',
                                 scaling = 'global')
        
        # Save data and labels
        # ---------------------------------------------------------
        if signal_collect is not None:
            
            if collect_in_one_array:

                # Get signals as array and add to total collection
                print(signal_collect.shape, len(label_collect))
                signal_collection = np.concatenate((signal_collection, signal_collect), axis=0)
                label_collection += label_collect

            else:
                if len(label_collect) > 1:
                #if label_collect is not None:
                    file = os.path.join(PATH_OUTPUT, "processed_data_" + filename[:-4] + ".npy")
                    np.save(file, signal_collect)

                    #filename = os.path.join(PATH_OUTPUT, "EEG_data_30ch_1s_corrected_metadata_ID"+ ID + ".csv")
                    file = os.path.join(PATH_OUTPUT, "processed_data_" + filename[:-4] + ".csv")

                    with open(file, 'w', newline='') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(label_collect)
                    csvFile.close()
            
            ID_collection += [ID] * len(label_collect) 
            metadata_collection.append((i, filename, signal_collection.shape[0]))


# In[48]:


signal_collect is None


# In[34]:


filename[:-4]


# In[32]:


signal_collection.shape, len(label_collection), len(ID_collection)


# In[57]:


metadata_collection[:10]


# In[55]:


print("Unique labels found in data:", list(set(label_collection)))


# We hence get a dataset of 39083 datapoints with known label.  
# Each datapoint consits of a 1-second EEG signal of 30 channels with a 500Hz sampling rate. Thus arrays with a size of 30 x 501. 

# # Save entire processed dataset:

# In[75]:


filename = os.path.join(PATH_OUTPUT, "EEG_data_30channels_1s_corrected.npy")
np.save(filename, signal_collection)

filename = os.path.join(PATH_OUTPUT, "EEG_data_30channels_1s_corrected_labels.npy")
np.save(filename, label_collection)

import csv
filename = os.path.join(PATH_OUTPUT, "EEG_data_30channels_1s_corrected_metadata.csv")

with open(filename, 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(metadata_collection)
csvFile.close()

