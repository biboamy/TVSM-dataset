'''
This script contains the dataloader for loading Netflix evaluation dataset
'''

import soundfile as sf
import random
import os
import sys
sys.path.append("./dataloader/")
from dataset_module import Dataset
'''
default dataset path:
    audio: /{root folder}/netflix_eval/audio
    labels: /{root folder}/netflix_eval/labels or /{root folder}/netflix_eval/labels_SBSMD
'''

class NetflixEvalDataset(Dataset):

    def __getitem__(self, index):
        if self.partition == 'train':
            index = index//self.n_sample
        audio_length = sf.info(os.path.join(self.data_path, 'audio', self.file_list[index]+'.wav')).frames
        if self.partition == 'train':
            start = random.randrange(0, audio_length - int(self.duration * self.sr - 1))
            end = start + self.duration * self.sr
            audio, sr = sf.read(os.path.join(self.data_path, 'audio', self.file_list[index]), start=start, stop=end)
            label = self._extract_activation_roll(self.file_list[index].replace('.wav', '.csv'), audio_length)
            l_start = int(start / self.hop_size)
            l_end = int(self.duration * self.sr / self.hop_size) + l_start
            label = label[:, l_start: l_end]
        elif self.partition == 'val' or self.partition == 'test':
            audio, sr = sf.read(os.path.join(self.data_path, 'audio', self.file_list[index]+'.wav'))
            label = self._extract_activation_roll(self.file_list[index]+'.csv', audio_length)
        else:
            raise
        if audio.shape[-1] == 2:
            audio = audio.sum(-1)/2
        return audio.T, label, self.file_list[index]


