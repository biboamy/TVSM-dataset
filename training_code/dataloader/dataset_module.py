'''
This script contains the dataloader module
'''

import torch.utils.data as data
import soundfile as sf
import random
import os
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, data_path, partition, duration=4, sr=16000, hop_size=1024, n_sample=1, features='mel'):
        '''
        :param data_path: Path to the dataset
        :param partition: 'train' for training; 'val' for validation; 'test' for testing
        :param duration: the length of the chunk, only useful for training (sec)
        :param sr: sampling rate
        :param hop_size: the hop size to compute the features are also used to construct the groundtruth labels
        :param n_sample: how many chunks to be taken for each audio
        :param features: name of the input features 
        '''
        self.partition = partition
        self.data_path = data_path
        self.duration = duration
        self.sr = sr
        self.n_sample = n_sample
        self.hop_size=hop_size
        self.file_list = np.load(os.path.join(data_path, partition+'.npy'))
        self.labels_name= 'labels'
        self.features = features

    def _extract_activation_roll(self, name, duration):
        '''
        extract activation matrix from csv timing
        :param name: file name
        :param duration: the length of the matrix
        :return: activation matrix (2 X length)
        '''
        activation_roll = np.zeros((2, int(duration/self.hop_size))) # music, speech
        file = open(os.path.join(self.data_path, self.labels_name, name.replace('wav', 'csv')), 'r')
        lines = file.readlines()
        for line in lines:
            start, end, label = line.strip().split('\t')
            start = int(float(start)*self.sr/self.hop_size)
            end = int(float(end)*self.sr/self.hop_size)
            if label == 'm':
                activation_roll[0, start:end] = 1
            if label == 's':
                activation_roll[1, start:end] = 1
        return activation_roll

    def __getitem__(self, index):
        try:
            if self.partition == 'train':
                index = index // self.n_sample

            audio_length = sf.info(os.path.join(self.data_path, 'audio', self.file_list[index])).frames
            if self.partition == 'train':
                start = random.randrange(0, audio_length - int(self.duration * self.sr - 1))
                end = start + self.duration * self.sr
                audio, sr = sf.read(os.path.join(self.data_path, 'audio', self.file_list[index]), start=start, stop=end)
                label = self._extract_activation_roll(self.file_list[index].replace('.wav', '.csv'), audio_length)
                l_start = int(start / self.hop_size)
                l_end = int(self.duration * self.sr / self.hop_size) + l_start
                label = label[:, l_start: l_end]
            elif self.partition == 'val' or self.partition == 'test':
                audio, sr = sf.read(os.path.join(self.data_path, 'audio', self.file_list[index]))
                label = self._extract_activation_roll(self.file_list[index].replace('.wav', '.csv'), audio_length)
            else:
                raise

            return audio.T, label, self.file_list[index]
        except Exception as e:
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)

    def __len__(self):
        if self.partition == 'train':
            return len(self.file_list) * self.n_sample
        elif self.partition == 'val' or self.partition == 'test':
            return len(self.file_list)
