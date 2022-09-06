'''
This script contains the dataloader for loading OpenBMAT dataset
'''

import soundfile as sf
import random
import os
import sys
from dataset_module import Dataset
from model import preprocess
import torch



class BMATDataset(Dataset):
	def __init__(self, data_path, partition, duration=4, sr=16000, hop_size=1024, n_sample=1, features='mel', n_fft=1024):
		self.preprocess = preprocess.MelSpec(sr, n_fft, hop_size)
		super(BMATDataset, self).__init__(data_path, partition, duration, sr, hop_size, n_sample, features)

	def __getitem__(self, index):
		try:
			if self.partition == 'train':
				index = index // self.n_sample

			audio_length = sf.info(os.path.join(self.data_path, 'audio_16000', self.file_list[index]+'.wav')).frames
			if self.partition == 'train':
				start = random.randrange(0, audio_length - int(self.duration * self.sr - 1))
				end = start + self.duration * self.sr
				audio, sr = sf.read(os.path.join(self.data_path, 'audio_16000', self.file_list[index]+'.wav'), start=start, stop=end)
				label = self._extract_activation_roll(self.file_list[index]+'.csv', audio_length)
				l_start = int(start / self.hop_size)
				l_end = int(self.duration * self.sr / self.hop_size) + l_start
				label = label[:, l_start: l_end]
			elif self.partition == 'val':
				audio, sr = sf.read(os.path.join(self.data_path, 'audio_16000', self.file_list[index]+'.wav'))
				label = self._extract_activation_roll(self.file_list[index]+'.csv', audio_length)
			else:
				audio, sr = sf.read(os.path.join(self.data_path, 'audio', self.file_list[index]+'.wav'))
				label = self._extract_activation_roll(self.file_list[index]+'.csv', audio_length)
				return audio.T, label, self.file_list[index]
			
			features = self.preprocess(torch.from_numpy(audio).float().unsqueeze(0)).squeeze()
			
			return features, label, self.file_list[index]

		except Exception as e:
			index = index - 1 if index > 0 else index + 1
			return self.__getitem__(index)