import glob
import os
import json
import numpy as np
import librosa
import soundfile as sf
import tqdm


# process the original labels to our csv version
def process_label(dir_path, des_path):
	if not os.path.exists(des_path):
		os.mkdir(des_path)

	with open(os.path.join(dir_path, 'annotations/json/MD_mapping.json')) as f:
		data = json.load(f)

	# ['agreement', 'annotations']
	for annotator in data['annotations'].keys():
		for _id in data['annotations'][annotator]:
			oup_data = open(os.path.join(des_path, f'{_id}.csv'), 'w')

			for num in data['annotations'][annotator][_id].keys():
				seg = data['annotations'][annotator][_id][num]	
				if seg['class'] == 'music':
					oup_data.write(str(seg['start'])+'\t'+str(seg['end'])+'\t'+'m'+'\n')
			oup_data.close()

#process_label('../../../data/OpenBMAT/', '../../../data/OpenBMAT/labels/')

def downsample_audio(dir_path, des_path):
	if not os.path.exists(des_path):
		os.mkdir(des_path)

	for file in tqdm.tqdm(os.listdir(dir_path)):
		data, sr = librosa.load(os.path.join(dir_path, file), sr=16000)
		print(data.shape, sr)
		sf.write(os.path.join(des_path, file), data, sr)

#downsample_audio('../../../data/OpenBMAT/audio/', '../../../data/OpenBMAT/audio_16000/')