import soundfile
import os
from sklearn.model_selection import train_test_split
import numpy as np
import glob
from tqdm import tqdm

# Calculate the dataset length
def calculate_dataset_length(dataset_path):
    accumulate_time = 0
    for file in tqdm(os.listdir(dataset_path)):
        if file.endswith('npy'):
            try:
                #audio_info = soundfile.info(os.path.join(dataset_path, file))
                #accumulate_time += (audio_info.duration)
                audio_info = np.load(os.path.join(dataset_path, file))
                accumulate_time += (audio_info.shape[1]/31.25)
            except: pass

    print(accumulate_time/60/60)
#calculate_dataset_length('../data/netflix_whole/mel_features/')

# split the dataset into training, validation, and test (optional)
def split_dataset(dataset_path, split_test=True):
    file_list = glob.glob(os.path.join(dataset_path, 'labels', '*')) #np.load('netflix/subsetID.npy')#glob.glob(os.path.join(dataset_path, 'audio', '*'))
    if split_test:
        dataset_len = len(file_list)
        train_files, test_files = train_test_split(file_list, test_size=0.1)
        train_files, val_files = train_test_split(train_files, test_size=0.11)
        train_files = [os.path.basename(file) for file in train_files]
        val_files = [os.path.basename(file) for file in val_files]
        test_files = [os.path.basename(file) for file in test_files]
        np.save(os.path.join(dataset_path, 'train.npy'), train_files)
        np.save(os.path.join(dataset_path, 'val.npy'), val_files)
        np.save(os.path.join(dataset_path, 'test.npy'), test_files)
        print(len(train_files), len(val_files), len(test_files))
    else:
        train_files, val_files = train_test_split(file_list, test_size=0.1)
        train_files = [os.path.basename(file).split('.')[0] for file in train_files]
        val_files = [os.path.basename(file).split('.')[0] for file in val_files]
        np.save(os.path.join(dataset_path, 'train.npy'), train_files)
        np.save(os.path.join(dataset_path, 'val.npy'), val_files)
        print(len(train_files), len(val_files))
#split_dataset('../../data/OpenBMAT/', False)

# resample the whole dataset into 22050 sampling rate
def resample_audio(dataset_path, des_path):
    import subprocess as sp
    for file in tqdm(os.listdir(dataset_path)):
        if not file.startswith('.'):
            ffmpeg_command = ['ffmpeg', '-i', os.path.join(dataset_path, file), '-ar', '22050', os.path.join(des_path, file)]
            print(ffmpeg_command)
            sp.call(ffmpeg_command)
#download_sample_audio('/data/dataset/avaspeech/audio/', '/data/dataset/avaspeech/audio_22050/')

# combine speech (labels_s) and music (labels_m) into training labels (labels)
def combine_labels(dir_path, com_path, des_path):
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    for file in os.listdir(dir_path):
        if not file.startswith('.'):
           if os.path.exists(os.path.join(com_path, file)):
                ori_data = open(os.path.join(dir_path, file), 'r')
                com_data = open(os.path.join(com_path, file), 'r')
                oup_data = open(os.path.join(des_path, file), 'w')
                for line in ori_data.readlines():
                    oup_data.write(line.replace(',', '\t'))
                for line in com_data.readlines():
                    if line.strip().split('\t')[-1] == 's':
                        oup_data.write(line)
                oup_data.close()

combine_labels('../../../AVASpeech_Music_Labels/music labels/', '../../../AVASpeech_Music_Labels/speech labels/', '../../../AVASpeech_Music_Labels/mix labels/')

def csv_to_npy(dir_path, des_path):
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    for file in tqdm(os.listdir(os.path.join(dir_path,'labels_raw'))):
        mel = np.load(os.path.join(dir_path, 'mel_features', file.replace('csv', 'npy')))
        activation_roll = np.zeros((2, int(mel.shape[1])))
        raw_labels = open(os.path.join(dir_path, 'labels_raw', file), 'r')
        lines = raw_labels.readlines()
        for line in lines:
            start, end, label = line.strip().split('\t')
            start = round((float(start))*16000/512)
            end = round((float(end))*16000/512)
            if start >= 0 and start < mel.shape[1]:
                if label == 'm':
                    activation_roll[0,start:end] = 1
                if label == 's':
                    activation_roll[1, start:end] = 1
        print(activation_roll.sum(1), activation_roll.shape[-1])
        np.save(os.path.join(des_path, file.replace('csv', 'npy')), activation_roll)

#csv_to_npy('../../data/netflix_whole/', '../../data/netflix_whole/labels/')