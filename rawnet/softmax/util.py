# -*- coding: utf-8 -*-
import librosa
import numpy as np
import random
import scipy
# from weidi's utils'

def load_data(path, sr=16000, time_len=3 ,mode='train'):

    if mode == 'train':
        file_len = librosa.get_duration(filename=path)
        # if get_duration of the wav from librosa.load, should consider the sr
        randomtime = random.uniform(0, file_len - time_len)
        # print("overall length is {0}, \n randomtime is {1}".format(file_len, randomtime))
        wav, sr_ret = librosa.load(path, sr, duration=time_len,
                                   offset=randomtime)

        # alternative but slower way
        # wav, sr_ret = librosa.load(vid_path, sr)
        # sr, wav = scipy.io.wavfile.read(path)
        # randtime = random.randint(0,len(wav)-sr * time_len)
        # wav = wav[randtime:randtime+sr * time_len]
        if len(wav) != sr*time_len:
            print("ERROR:the input data to the CNN have different forms")
    else:
        wav, sr_ret = librosa.load(path, sr)  # just simply load the data, e.g. test data
        # sr, wav = scipy.io.wavfile.read(path)
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(wav, 0, keepdims=True)
    std = np.std(wav, 0, keepdims=True)
    return (wav - mu) / (std + 1e-5)

def load_data_mfcc(path, sr=16000, mode='train'):
    time_len = 3
    if mode == 'train':
        file_len = librosa.get_duration(filename=path)
        # if get_duration of the wav from librosa.load, should consider the sr
        randomtime = random.uniform(0, file_len - time_len)
        # print("overall length is {0}, \n randomtime is {1}".format(file_len, randomtime))
        wav, sr_ret = librosa.load(path, sr, duration=time_len,
                                   offset=randomtime)
        # alternative but slower way
        # wav, sr_ret = librosa.load(vid_path, sr)
        # randtime = random.randint(0,len(wav)-sr * time_len)
        # new_wav = wav[randtime:randtime+sr * time_len]
        if len(wav) != sr*time_len:
            print("ERROR:the input data to the CNN have different forms")
    else:
        wav, sr_ret = librosa.load(path, sr)  # just simply load the data, e.g. test data
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(wav, 0, keepdims=True)
    std = np.std(wav, 0, keepdims=True)
    return (wav - mu) / (std + 1e-5)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]