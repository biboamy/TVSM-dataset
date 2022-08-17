import torch.utils.data as data
import soundfile as sf
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import librosa
import librosa.display


class NetflixWholeDataset(data.Dataset):
    def __init__(self, data_path, partition, duration=4, sr=16000, hop_size=1024, n_sample=1, features=None):
        self.partition = partition
        self.data_path = data_path
        self.duration = duration
        self.sr = sr
        self.n_sample = n_sample
        self.hop_size = hop_size
        self.file_list = np.load(os.path.join(data_path, partition+'.npy'))
        self.features = features

    def __getitem__(self, index):
        try:
            chunk_size = int(self.duration * self.sr / self.hop_size)
            if self.partition == 'train':
                feature = np.load(os.path.join(self.data_path, self.features+'_features', self.file_list[index]+'.npy')) # time x features
                label = np.load(os.path.join(self.data_path, 'labels', self.file_list[index]+'.npy')) #np.repeat(np.load(os.path.join(self.data_path, 'labels', self.file_list[index]+'.npy')), 2, axis=1)
                features_list, label_list = [], []
                for i in range(16):
                    start = random.randrange(0, feature.shape[1] - chunk_size - 20)
                    end = start + chunk_size
                    
                    features_list.append(feature[:, start: end])
                    label_list.append(label[:, start: end])
                features = np.array(features_list)
                labels = np.array(label_list)

            elif self.partition == 'val' or self.partition == 'test':
                features = np.load(os.path.join(self.data_path, self.features+'_features', self.file_list[index]+'.npy'))[:, :9375]
                labels = np.load(os.path.join(self.data_path, 'labels', self.file_list[index]+'.npy'))[:, :9375]

                #nums = features.shape[-1] // chunk_size 
                #features_list, label_list = [], []
                #for num in range(nums):
                #    features_list.append()
            else:
                raise "Should be in [train, val, test]"

            return features, labels, self.file_list[index]
        except Exception as e:
            #print(e)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)

    def __len__(self):
        if self.partition == 'train':
            return len(self.file_list) 
        elif self.partition == 'val' or self.partition == 'test':
            return len(self.file_list)
