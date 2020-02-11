import torch
import torchvision
import numpy as np
import util as ut
import torch.utils.data as data
import pdb

class DataLoader(data.Dataset):
    'load the data from the processed dataset'


    def __init__(self, list_IDs, labels, dim, mp_pooler, augmentation=True, batch_size=32, nfft=512, spec_len=250,
                     win_length=400, sampling_rate=16000, hop_length=160, n_classes=5994, shuffle=True, normalize=True):

            self.dim = dim
            self.nfft = nfft
            self.sr = sampling_rate
            self.spec_len = spec_len
            self.normalize = normalize
            self.mp_pooler = mp_pooler
            self.win_length = win_length
            self.hop_length = hop_length
            self.labels = labels
            self.shuffle = shuffle
            self.list_IDs = list_IDs  # addresses of the data
            self.n_classes = n_classes
            self.batch_size = batch_size
            self.augmentation = augmentation
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle:
                np.random.shuffle(self.indexes)

    def __len__(self):
        #shows the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the data of  the new batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # pdb.set_trace()
        # Find list of IDs which is the address of the audios
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data  X= input data, y= one-hot encode tensor
        X, y = self.__data_generation_mp(list_IDs_temp, indexes)
        # pdb.set_trace()
        return X, y

    def __data_generation_mp(self, list_IDs_temp, indexes):
        X0 = [self.mp_pooler.apply_async(ut.load_data,
                                        args=(ID, self.sr,
                                               )) for ID in list_IDs_temp]
        X = np.expand_dims(np.array([p.get() for p in X0]), -1)

        y = self.labels[indexes]
        # pdb.set_trace()
        return X, y

# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return np.eye(num_classes, dtype='uint8')[y]

if __name__ == '__main__':
    print(DataLoader())