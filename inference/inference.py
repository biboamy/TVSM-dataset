import os
import numpy as np
import librosa
import torch
import torchvision.transforms as T
import torchaudio
from pcen import PCENTransform
import tqdm
import CRNN
import argparse
import csv

sr = 16000
n_fft = 1024
hop_size = 512
n_features = 128
duration = 20

music_threshold = 0.5
speech_threshold = 0.5
here = os.path.dirname(os.path.abspath(__file__))
pseudo_model_path = os.path.join(here, 'models', 'TVSM-pseudo', 'epoch=28-step=67192.ckpt.torch.pt')


def model_creator():
    model = CRNN.CRNN()
    return model


# pseudo_model_path = 'abc'
class SMDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = model_creator()
        self.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.pcen_transform = T.Compose([
            torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_size).to(self.device),
            PCENTransform().to(self.device)
        ])
        print(f'Finish loading SMDetector device: {self.device}')

    def load_from_checkpoint(self, model_path):
        print(f'Loading model from {model_path}')
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)

    def predict_audio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        y = np.expand_dims(y, 0)
        audio = torch.from_numpy(y).float()
        audio = audio.to(self.device)
        audio_pcen_data = self.pcen_transform(audio)
        c_size = int(sr / hop_size * duration)
        n_chunk = int(np.ceil(audio_pcen_data.shape[-1] / c_size))
        est_label = []
        with torch.inference_mode():
            for i in range(n_chunk):
                chunk_data = audio_pcen_data[..., i * c_size:(i + 1) * c_size]
                la = self.model(chunk_data).detach().cpu()
                est_label.append(la)
        est_label = torch.cat(est_label, -1)
        est_label = torch.sigmoid(est_label)
        est_label = torch.max_pool1d(est_label, 6, 6)
        frame_time = 1 / ((sr / hop_size) / 6)
        audio_label_results = []
        est_label = est_label.detach().cpu().numpy()[0]
        for i, frame in enumerate(est_label.T):
            start_time_s = str(frame_time * i)
            end_time_s = str(frame_time * (i + 1))
            music_prob = round(float(frame[0]), 2)
            speech_prob = round(float(frame[1]), 2)
            result = {'start_time_s': start_time_s,
                      'end_time_s': end_time_s,
                      'music_prob': music_prob,
                      'speech_prob': speech_prob}
            audio_label_results.append(result)
        return audio_label_results


def export_prob_result(filename, result):
    with open(filename, 'w', ) as csvfile:
        fieldnames = ['start_time_s', 'end_time_s', 'music_prob', 'speech_prob']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in result:
            writer.writerow(r)


def export_result(filename, result):
    with open(filename, 'w') as csvfile:
        for r in result:
            print(r)
            if r['music_prob'] > music_threshold:
                csvfile.write(r['start_time_s'] + '\t' + r['end_time_s'] + '\t' + 'm' + '\n')
            if r['speech_prob'] == speech_threshold:
                csvfile.write(r['start_time_s'] + '\t' + r['end_time_s'] + '\t' + 's' + '\n')


def main(audio_path, output_dir):
    if not os.path.exists(audio_path):
        print('No such file or directory: ', audio_path)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    smd = SMDetector(pseudo_model_path)

    if os.path.isdir(audio_path):
        all_files = []
        for root, dirs, files in os.walk(audio_path):
            for file in files:
                full_path = os.path.join(root, file)
                all_files.append(full_path)

        for full_path in tqdm.tqdm(all_files):
            file_result = smd.predict_audio(full_path)
            result_csv_filename = os.path.join(output_dir, os.path.basename(full_path) + '.csv')
            export_result(result_csv_filename, file_result)

    else:
        file_result = smd.predict_audio(audio_path)
        result_csv_filename = os.path.join(output_dir, os.path.basename(audio_path) + '.csv')
        export_result(result_csv_filename, file_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TVSM Inference', description='TVSM Inference')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'csv_prob'])
    args = parser.parse_args()
    main(args.audio_path, args.output_dir)
