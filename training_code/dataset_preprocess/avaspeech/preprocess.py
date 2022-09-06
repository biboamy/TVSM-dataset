import csv
import os
from tqdm import tqdm

# Extract audio id from the label list
def generate_dataset_id(ava_speech_label_path, des_file):
    oup_file = open(des_file, "w")
    names = []
    with open(ava_speech_label_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0] not in names:
                names.append(row[0])
    for name in names:
        oup_file.write(name+'\n')
    oup_file.close()
#generate_dataset_id('ava_speech_labels_v1.csv', "ava_speech_file_names_v1.txt")

# generate speech labels for each audio id
def generate_label(ava_speech_label_path, des_path):
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    labels = {}
    with open(ava_speech_label_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            start = str(float(row[1]) - 900)
            end = str(float(row[2]) - 900)
            if row[0] not in labels:
                labels[row[0]] = []
            if row[3] == 'SPEECH_WITH_NOISE':
                labels[row[0]].append(start + '\t' + end + '\t' + 's')
            if row[3] == 'SPEECH_WITH_MUSIC':
                labels[row[0]].append(start + '\t' + end + '\t' + 's')
                labels[row[0]].append(start + '\t' + end + '\t' + 'm')
            if row[3] == 'CLEAN_SPEECH':
                labels[row[0]].append(start + '\t' + end + '\t' + 's')
    for name in labels.keys():
        oup_file = open(os.path.join(des_path, name+'.csv'), "w")
        for label in labels[name]:
            oup_file.write(label + '\n')
        oup_file.close()

#generate_label('/root/avaspeech/ava_speech_labels_v1.csv', '/root/avaspeech/labels_s/')

# download script
# download.sh

# chunck audio into 15 mins used in the dataset
def chunk_audio(dir_path, des_path):
    import subprocess
    for file in tqdm(os.listdir(dir_path)):
        if not file.startswith('.'):
            #cmd = ['ffmpeg', '-ss', '900', '-t', '900', '-i', os.path.join(dir_path, file), '-ar', '22050', os.path.join(dir_path, os.path.splitext(os.path.splitext(file)[0])[0]+'.wav')]
            cmd = ['ffmpeg', '-ss', '900', '-t', '900', '-i', os.path.join(dir_path, file), os.path.join(des_path, file)]
            process = subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#chunk_audio('/root/avaspeech/audio_ori/', '/root/avaspeech/audio/')