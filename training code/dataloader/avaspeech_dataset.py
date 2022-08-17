'''
This script contains the dataloader for loading AVASpeech dataset
'''

import soundfile as sf
import random
import os
import sys
sys.path.append("./dataloader/")
from dataset_module import Dataset
'''
default dataset path:
    audio: /{root folder}/avaspeech/audio
    labels: /{root folder}/avaspeech/labels or /{root folder}/avaspeech/labels_SBSMD
'''

class AVASpeechDataset(Dataset):

    def __getitem__(self, index):
        if self.partition == 'train':
            index = index//self.n_sample

        audio_length = sf.info(os.path.join(self.data_path, 'audio', self.file_list[index])).frames
        if self.partition == 'train':
            start = random.randrange(0, audio_length-int(audio_length-self.duration*self.sr-1))
            end = start+self.duration*self.sr
            audio, sr = sf.read(os.path.join(self.data_path, 'audio', self.file_list[index]), start=start, stop=end)
            label = self._extract_activation_roll(self.file_list[index], audio_length)
            l_start = int(start/self.hop_size)
            l_end = int(self.duration*self.sr/self.hop_size) + l_start
            label = label[:, l_start: l_end]
        elif self.partition == 'val' or self.partition == 'test':
            audio, sr = sf.read(os.path.join(self.data_path, 'audio', self.file_list[index]))
            label = self._extract_activation_roll(self.file_list[index], audio_length)
        else:
            raise 

        if audio.shape[-1] == 2:
            audio = audio.sum(-1) / 2
        return audio.T, label, self.file_list[index]
