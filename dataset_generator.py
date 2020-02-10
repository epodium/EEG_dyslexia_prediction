# Import libraries
from tensorflow.keras.utils import Sequence
import numpy as np
import os
import csv


class DataGenerator(Sequence):
    """Generates data for loading (preprocessed) EEG timeseries data.
    Create batches for training or prediction from given folders and filenames.
    
    """
    def __init__(self, 
                 list_IDs,
                 main_labels,
                 path_data,
                 filenames,
                 data_path, 
                 to_fit=True, 
                 n_average = 30,
                 batch_size=32,
                 iter_per_epoch = 2,
                 up_sampling = True,
                 n_timepoints = 501,
                 n_channels=30, 
                 n_classes=1, 
                 shuffle=True):
        """Initialization
        
        Args:
        --------
        list_IDs: 
            list of all filename/label ids to use in the generator
        main_labels: 
            list of all main labels.
        path_data: str
            Foldername for all npy and csv files...
        filenames: 
            list of image filenames (file names)
        data_path: 
            path to data directory
        to_fit: 
            True to return X and y, False to return X only
        n_average: int
            Number of EEG/time series epochs to average.
        batch_size: 
            batch size at each iteration
        iter_per_epoch: int
            Number of iterations over all data points within one epoch.
        up_sampling: bool
            If true, create equal amounts of data for all main labels.
        n_timepoints: int
            Timepoint dimension of data.
        n_channels: 
            number of input channels
        n_classes: 
            number of output channels ?
        shuffle: 
            True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.main_labels = main_labels
        self.path_data = path_data
        self.filenames = filenames
        self.data_path = data_path
        self.to_fit = to_fit
        self.n_average = n_average
        self.batch_size = batch_size
        self.iter_per_epoch = iter_per_epoch
        self.up_sampling = up_sampling
        self.n_timepoints = n_timepoints
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch
        
        return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data
        
        Args:
        --------
        index: int
            index of the batch
        
        return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[int(k)] for k in indexes]

        # Generate data
        X, y = self.generate_data(list_IDs_temp)

        if self.to_fit:
            return X, y
        else:
            return X


    def on_epoch_end(self):
        """Updates indexes after each epoch.
        Takes care of up-sampling and sampling frequency per "epoch".
        """
      
        idx_labels = []
        label_count = []
        
        # Up-sampling
        if self.up_sampling:
            labels_unique = list(set(self.main_labels))
            for label in labels_unique:
                idx = np.where(np.array(self.main_labels) == label)[0]
                idx_labels.append(idx)
                label_count.append(len(idx))
             
            idx_upsampled = np.zeros((0))    
            for i in range(len(labels_unique)):
                up_sample_factor = self.iter_per_epoch * max(label_count)/label_count[i]
                idx_upsampled = np.concatenate((idx_upsampled, np.tile(idx_labels[i], int(up_sample_factor // 1))), 
                                               axis = 0)
                idx_upsampled = np.concatenate((idx_upsampled, np.random.choice(idx_labels[i], int(label_count[i] * up_sample_factor % 1), replace=True)),  
                                               axis = 0)
            self.indexes = idx_upsampled
            
        else:
            # No upsampling
            idx_base = np.arange(len(self.list_IDs))
            idx_epoch = np.tile(idx_base, self.iter_per_epoch)   

            self.indexes = idx_epoch
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
   
    
    def generate_data(self, list_IDs_temp):
        """Generates data containing batch_size averaged time series.
        
        Args:
        -------
        list_IDs_temp: list
            list of label ids to load
        
        return: batch of averaged time series
        """
        X_data = np.zeros((0, self.n_channels, self.n_timepoints))
        y_data = []
        
         # Generate data
        for i, ID in enumerate(list_IDs_temp):
            filename = self.filenames[ID]
            data_signal = self.load_signal(self.path_data, filename + '.npy')
            data_labels = self.load_labels(self.path_data, filename + '.csv')
            
            X, y = self.create_averaged_epoch(data_signal, 
                                              data_labels)
            
            X_data = np.concatenate((X_data, X), axis=0)
            y_data += y
        
        if self.shuffle:
            idx = np.arange(len(y_data))
            np.random.shuffle(idx)
            X_data = X_data[idx, :, :]
            y_data = [y_data[i] for i in idx]
        
        return X_data, y_data
    

    def create_averaged_epoch(self,
                              data_signal, 
                             data_labels):
        """ 
        Function to create averages of self.n_average epochs.
        Will create one averaged epoch per found unique label from self.n_average random epochs.
        
        Args:
        
        
        """
        # Create new data collection:
        X_data = np.zeros((0, self.n_channels, self.n_timepoints))
        y_data = []
        
        categories_found = list(set(data_labels))
    
        idx_cat = []
        for cat in categories_found:
            idx = np.where(np.array(data_labels) == cat)[0]
            idx_cat.append(idx)
            
            if len(idx) >= self.n_average:        
                select = np.random.choice(idx, self.n_average, replace=False)
            elif len(idx) >= self.n_average/2: 
                print("Found only", len(idx), " epochs and will take those!")
                signal_averaged = np.mean(data_signal[idx,:,:], axis=0)
                break
            else:
                break
                
            signal_averaged = np.mean(data_signal[select,:,:], axis=0)
            X_data = np.concatenate([X_data, np.expand_dims(signal_averaged, axis=0)], axis=0)
            y_data.append(cat)
    
        return X_data, y_data


    def load_signal(self, 
                    PATH,
                    filename):
        """Load EEG signal from one person.
        
        Args:
        -------
        filename: str
            filename...
        PATH: str
            path name to file to load
        
        return: loaded array
        """
        return np.load(os.path.join(PATH, filename))

        
    def load_labels(self,
                    PATH, 
                    filename):
        metadata = []
        filename = os.path.join(PATH, filename)
        with open(filename, 'r') as readFile:
            reader = csv.reader(readFile, delimiter=',')
            for row in reader:
                #if len(row) > 0:
                metadata.append(row)
        readFile.close()
        
        return metadata[0]
     
        