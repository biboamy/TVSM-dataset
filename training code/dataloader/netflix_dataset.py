import torch.utils.data as data
import soundfile as sf
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import librosa
import librosa.display


class NetflixDataset(data.Dataset):
    def __init__(self, data_path, partition, duration=4, sr=16000, hop_size=1024, n_sample=1, features=None):
        self.partition = partition
        self.data_path = data_path
        self.duration = duration
        self.sr = sr
        self.n_sample = n_sample
        self.hop_size = hop_size
        self.file_list = np.load(os.path.join(data_path, partition+'.npy'))
        self.features = features

    def _extract_activation_roll(self, name, duration):
        activation_roll = np.zeros((2, int(duration)))
        file = open(os.path.join(self.data_path, 'labels', name), 'r')
        lines = file.readlines()
        for line in lines:
            start, end, label = line.strip().split('\t')
            ident = 300
            #if not self.labels_name == 'labels' and label == 'm':
            #    ident = 0
            start = int((float(start)-ident)*self.sr/self.hop_size)
            end = int((float(end)-ident)*self.sr/self.hop_size)
            if start >= 0 and start < duration:
                if label == 'm':
                    activation_roll[0,start:end] = 1
                if label == 's':
                    activation_roll[1, start:end] = 1

        return activation_roll

    def __getitem__(self, index):
        try:
            if self.partition == 'train':
                index = index//self.n_sample

            if self.partition == 'train':
                features = np.load(os.path.join(self.data_path, self.features+'_features', self.file_list[index]+'.npy')) # time x features
                chunk_size = int(self.duration * self.sr / self.hop_size)
                start = random.randrange(0, features.shape[1] - chunk_size)
                end = start + chunk_size
                label = self._extract_activation_roll(self.file_list[index] + '.csv', features.shape[1])
                features = features[:, start: end]
                label = label[:, start: end]

            elif self.partition == 'val' or self.partition == 'test':
                features = np.load(os.path.join(self.data_path, self.features+'_features', self.file_list[index]+'.npy'))
                label = self._extract_activation_roll(self.file_list[index]+'.csv', features.shape[1])
            else:
                raise "Should be in [train, val, test]"
            return features, label, self.file_list[index]
        except Exception as e:
            print(e)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)

    def __len__(self):
        if self.partition == 'train':
            return len(self.file_list) * self.n_sample
        elif self.partition == 'val' or self.partition == 'test':
            return len(self.file_list)
