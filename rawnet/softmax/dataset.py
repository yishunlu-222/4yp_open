import librosa
import numpy as np
import random
import os
import torch.utils.data as data
import util as ut

class voxceleb_dataset(data.Dataset):
    def __init__(self, list_IDs, labels,sr=12000, time_len=3, mode='train'):
        self.paths =list_IDs
        self.mode = mode
        self.labels = labels
        self.sr= sr
        self.time_len = time_len

    def __getitem__(self, index):
        path = self.paths[index]
        audio  = ut.load_data(path, self.sr,self.time_len, self.mode)
        label = self.labels[index]
        return audio, label
    def __len__(self):
        return len(self.paths)
