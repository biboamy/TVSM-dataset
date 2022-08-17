import os
import numpy as np

# process the original labels into our csv version
def refine_labels(dir_path, des_path):
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    for file in os.listdir(dir_path):
        if not file.startswith('.'):
            oup_file = open(os.path.join(des_path, file.replace('txt','csv')), "w")
            fp = open(os.path.join(dir_path, file), 'r')
            for line in fp.readlines():
                data = line.strip().split('\t')
                if data[2] == 'music':
                    label = 'm'
                if data[2] == 'speech':
                    label = 's'
                oup_file.write(data[0]+'\t'+data[1]+'\t'+label+'\n')
            oup_file.close()
#refine_labels('/data/dataset/muspeak/labels_raw/', '/data/dataset/muspeak/labels/')

# save all the files into "test.npy"
def generate_filelist(dir_path):
    file_list = []
    for file in os.listdir(dir_path):
        if not file.startswith('.'):
            file_list.append(file)
    print(len(file_list))
    np.save('/data/dataset/muspeak/test.npy', file_list)

#generate_filelist('/data/dataset/muspeak/audio/')
