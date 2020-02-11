# -*- coding: utf-8 -*-
import librosa
import numpy as np
import random

# from weidi's utils'
# def load_wav(vid_path, sr, mode='train'):
#     wav, sr_ret = librosa.load(vid_path, sr=sr)
#     assert sr_ret == sr   #if sr_ret != sr, break
#     if mode == 'train':
#         extended_wav = np.append(wav, wav)
#         if np.random.random() < 0.3:
#             extended_wav = extended_wav[::-1]
#         return extended_wav
#     else:
#         extended_wav = np.append(wav, wav[::-1])
#         return extended_wav


# def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
#     linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
#     return linear.T


# def load_data(path, sr=16000, spec_len=300, mode='train'):
#     wav = load_wav(path, sr=sr, mode=mode)
#     # linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
#     mag, _ = librosa.magphase(linear_spect)  # magnitude
#     mag_T = mag.T
#     freq, time = mag_T.shape
#     if mode == 'train':
#         randtime = np.random.randint(0, time-spec_len)
#         spec_mag = mag_T[:, randtime:randtime+spec_len]
#     else:
#         spec_mag = mag_T
#     # preprocessing, subtract mean, divided by time-wise var
#     mu = np.mean(spec_mag, 0, keepdims=True)
#     std = np.std(spec_mag, 0, keepdims=True)
#     return (spec_mag - mu) / (std + 1e-5)

def load_data(path, sr=16000, mode='train'):
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