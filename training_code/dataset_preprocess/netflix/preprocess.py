import numpy as np
import os
import subprocess as sp
import matplotlib.pyplot as plt
import tqdm
import soundfile as sf
import json
import sys
sys.path.append('../../')
from model import preprocess


# create center and other channel
def LtRt_creator(dir_path, des_path):
    import soundfile as sf
    """
    Convert 5.1 channel to LtRt, and save the Lt, Rt, center and others channels (wav).
    (Note LtRt(left, right) here means LoRo)
    input_path: wav file path
    output_dir: directory to save all the channels
    """
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    for file in tqdm.tqdm(os.listdir(dir_path)):
        input_path = os.path.join(dir_path, file)
        y, sr = sf.read(input_path)
        # ==== channel layout for channel mapping 10 = {L,R,C,LFE,Ls,Rs}
        L = y[:, 0]
        R = y[:, 1]
        C = y[:, 2]
        LFE = y[:, 3]
        Ls = y[:, 4]
        Rs = y[:, 5]
        # ==== simple downmix
        Lt = L + 0.707 * Ls
        Rt = R + 0.707 * Rs
        others_downmix = np.divide(Lt + Rt, 2)
        sf.write(os.path.join(des_path, file.replace('.wav', '_center.wav')), C, sr)
        sf.write(os.path.join(des_path, file.replace('.wav', '_others.wav')), others_downmix, sr)
        #sf.write(os.path.join(output_dir, "Lt.wav"), Lt, sr)
        #sf.write(os.path.join(output_dir, "Rt.wav"), Rt, sr)

#LtRt_creator('audio/', 'audio_downmix/')

# extract vgg features by using the pre-trained model
def extract_vgg_features(dir_path, des_path):
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    import torch, librosa, scipy
    vgg = torch.hub.load('harritaylor/torchvggish', 'vggish')#.to('cuda')
    #vgg.preprocess = False
    vgg.eval()
    transform = vgg #preprocess.vgg_mel()#.to('cuda')
    for file in tqdm.tqdm(os.listdir(dir_path)):
        if not os.path.exists(des_path + file.replace('.wav', '.npy')):
            if not file.startswith('.'):#not os.path.exists('/data/dataset/netflix/vgg_features/'+file.replace('.wav', '.npy')) and :
                #y, sr = sf.read(os.path.join(dir_path, file))
                '''
                L = y[:, 0]
                R = y[:, 1]
                C = y[:, 2]
                Ls = y[:, 4]
                Rs = y[:, 5]
                Lt = L + 0.707 * Ls
                Rt = R + 0.707 * Rs
                others_downmix = np.divide(Lt + Rt, 2)
                audio = others_downmix + 0.707 * C #np.concatenate([others_downmix[None,...], 0.707 * C[None,...]], 0)
                '''
                #x = torch.from_numpy(audio).float()#.to('cuda')
                #x = transform(x.unsqueeze(0).unsqueeze(0)).squeeze()
                x = transform(dir_path+file)

                print(x.shape, des_path + file.replace('.wav', '.npy'))
                np.save(des_path + file.replace('.wav', '.npy'), x.detach().cpu().numpy())

# (channel) x frames x bins
#extract_vgg_features('netflix/data/netflix_eval/audio_16000/', 'netflix/data/netflix_eval/vgg_features/')

# extract pcen features
def extract_pcen_features(dir_path, des_path):
    import pcen
    import torch
    transform = pcen.StreamingPCENTransform(n_mels=128, n_fft=1024, hop_length=512, trainable=False).to('cuda')

    if not os.path.exists(des_path):
        os.makedirs(des_path)

    for file in tqdm.tqdm(os.listdir(dir_path)[:]):
        if not file.startswith('.') and not os.path.exists(des_path+file.replace('.wav', '.npy')):
            y, sr = sf.read(os.path.join(dir_path, file))
            L = y[:, 0]
            R = y[:, 1]
            C = y[:, 2]
            Ls = y[:, 4]
            Rs = y[:, 5]
            Lt = L + 0.707 * Ls
            Rt = R + 0.707 * Rs
            others_downmix = np.divide(Lt + Rt, 2)
            audio = others_downmix + 0.707 * C #np.concatenate([others_downmix[None,...], 0.707 * C[None,...]], 0)

            data = []
            chunk_size = 16000 * 960
            length = sf.info(os.path.join(dir_path, file)).frames
            for chunk in range(int(length // chunk_size) + 1):
                data.append(transform(torch.from_numpy(audio[chunk*chunk_size:(chunk+1)*chunk_size]).to('cuda').float().unsqueeze(0)).squeeze())
            data = torch.cat(data, 0).T
            print(data.shape)
            np.save(des_path + file.replace('.wav', '.npy'), data.detach().cpu().numpy())

#extract_pcen_features('/root/restset/audio_16000/', '/root/restset/pcen/')

# extract mel spectrogram
def extract_mel_features(dir_path, des_path):
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    import torch
    transform = preprocess.MelSpec(16000, 1024, 512)#.to('cuda')

    for file in tqdm.tqdm(os.listdir(dir_path)):
        if not file.startswith('.') and not os.path.exists(des_path+file.replace('.wav', '.npy')):
            y, sr = sf.read(os.path.join(dir_path, file))

            '''
            L = y[:, 0]
            R = y[:, 1]
            C = y[:, 2]
            Ls = y[:, 4]
            Rs = y[:, 5]
            Lt = L + 0.707 * Ls
            Rt = R + 0.707 * Rs
            others_downmix = np.divide(Lt + Rt, 2)
            audio = others_downmix + 0.707 * C #np.concatenate([others_downmix[None,...], 0.707 * C[None,...]], 0)
            '''
            audio = y
            data = transform(torch.from_numpy(audio).float().unsqueeze(0)).squeeze()

            #data_s = transform(torch.from_numpy(C).to('cuda').float().unsqueeze(0))
            #data_m = transform(torch.from_numpy(others_downmix).to('cuda').float().unsqueeze(0))
            #data = torch.cat([data_m, data_s], 0)

            print(data.shape)
            np.save(des_path + file.replace('.wav', '.npy'), data.detach().cpu().numpy())
            
# bins x frames
#extract_mel_features('netflix/data/netflix_eval/audio_16000/', 'netflix/data/netflix_eval/mel_features/')

# chunk numpy array into 20 seconds
def chunk_files(dir_path, des_path):
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    for file in tqdm.tqdm(os.listdir(dir_path)):
        length = sf.info('/root/wholeset/audio_16000/'+file.replace('.npy', '.wav')).frames // (16000 * 20)
        chunk_size = 625 #/ 2
        data = np.load(os.path.join(dir_path, file))
        for chunk in range(length):
            if True:
            #if not os.path.exists(os.path.join(des_path, file.split('.')[0]+'_'+str(chunk)+'.npy')):

                start = int(np.round(chunk*chunk_size))#/0.96)
                end = start + 625 #(chunk+1)*chunk_size #start + 20#
                save_data = data[:, start:end]
                name = os.path.join(des_path, file.split('.')[0]+'_'+str(chunk)+'.npy')
                print(data.shape, save_data.shape, name, start, end)
                np.save(name, save_data)

#chunk_files('/root/wholeset/labels_e/', '/data/dataset/labels/')

def extract_mfcc():
    import librosa, soundfile

    files = os.listdir('/Volumes/Amy Volume/netflix/data/netflix_eval/mel_features/')
    for file in files:
        mel = np.load('/Volumes/Amy Volume/netflix/data/netflix_eval/mel_features/'+file)
        '''
        y, sr = soundfile.read('/Volumes/Amy Volume/netflix/data/netflix data/subset/audio_16000/'+file.replace('npy', 'wav'))
        L = y[:, 0]
        R = y[:, 1]
        C = y[:, 2]
        Ls = y[:, 4]
        Rs = y[:, 5]
        Lt = L + 0.707 * Ls
        Rt = R + 0.707 * Rs
        others_downmix = np.divide(Lt + Rt, 2)
        audio = others_downmix + 0.707 * C 
        _mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=1024, norm=None, htk=True)
        _librosa_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(_mel))
        '''
        librosa_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel))
        np.save('/Volumes/Amy Volume/netflix/data/netflix_eval/mfcc/'+file, librosa_mfcc)

        print(librosa_mfcc.shape)
#extract_mfcc()
