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
from utils import mono_check

# Audio / feature extraction parameters
# sr: target sample rate (Hz). Audio is loaded/resampled to this rate and
#     MelSpectrogram is created with this sample rate. Affects timing math.
sr = 16000
# n_fft: FFT window size in samples for STFT. Larger -> better freq. resolution,
#        worse time resolution.
n_fft = 1024
# hop_size: hop length (stride) in samples between adjacent STFT frames.
#           Determines frames-per-second = sr / hop_size.
hop_size = 512
# n_features: number of mel bins (n_mels) for the MelSpectrogram. Should match
#             the model's expected input channel dimension.
n_features = 128
# duration: length in seconds of each chunk fed to the model. Used to compute
#           c_size (frames per chunk) = int((sr / hop_size) * duration).
duration = 20

# Post-processing thresholds
# music_threshold / speech_threshold: probability cutoffs used when exporting
# labels. If a frame's probability exceeds the threshold, the corresponding
# label (m or s) is written to the output CSV.
music_threshold = 0.5
speech_threshold = 0.5

# Path helpers
# here: directory containing this script. pseudo_model_path: default checkpoint
# file used to initialize the model if no other path is provided.
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
        # Build feature transform pipeline using the global parameters:
        # - MelSpectrogram expects audio sampled at `sr` and uses `n_fft` and
        #   `hop_length` to compute STFT/mel frames. The resulting mel bins
        #   (n_mels) should match the model's expected input channels
        #   (n_features).
        # - PCENTransform is applied after the mel spectrogram to normalize
        #   and compress the spectral features.
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
        # Load audio and resample to the target sample rate `sr` (module-level).
        # Using librosa with sr=sr returns audio resampled to that rate. We
        # capture the returned sample rate in `loaded_sr` to avoid shadowing
        # the module-level `sr` constant.
        y, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
        y = np.expand_dims(y, 0)
        audio = torch.from_numpy(y).float()
        audio = audio.to(self.device)
        audio = mono_check(audio)

        # If the audio's sampling rate doesn't match the target `sr`,
        # resample the tensor to `sr` so the MelSpectrogram transform (which
        # was created with `sr`) receives audio at the expected rate.
        if loaded_sr != sr:
            resample = torchaudio.transforms.Resample(int(loaded_sr), sr)
            audio = resample(audio)

        # audio_pcen_data shape: (batch, n_mels, time_frames). The time axis
        # length (time_frames) is used to split the audio into chunks of
        # `duration` seconds for inference.
        audio_pcen_data = self.pcen_transform(audio)
        # c_size: number of mel feature frames that correspond to `duration`
        # seconds. frames_per_second = sr / hop_size, so c_size = frames_per_second * duration
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
        # Temporal max-pooling reduces the frame resolution by a factor of 6
        # (kernel=6, stride=6). This must be consistent with how the model
        # was trained/architected — changing the model's temporal pooling
        # requires updating this factor.
        est_label = torch.max_pool1d(est_label, 6, 6)
        # frame_time: seconds represented by each pooled frame. Derived from
        # hop_size, sr and the pooling factor (6): pooled_fps = (sr / hop_size) / 6
        # so frame_time = 1 / pooled_fps
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


def export_result(filename, result, format_type='csv'):
    # Export labels using the configured thresholds. For `csv` each frame
    # will be written as a line with start/end times and label if its
    # probability exceeds the corresponding threshold.
    if format_type == 'csv':
        with open(filename, 'w') as csvfile:
            for r in result:
                # Use music_threshold / speech_threshold to decide whether to
                # write a music/speech label for this frame interval.
                if r['music_prob'] > music_threshold:
                    csvfile.write(r['start_time_s'] + '\t' + r['end_time_s'] + '\t' + 'm' + '\n')
                if r['speech_prob'] > speech_threshold:
                    csvfile.write(r['start_time_s'] + '\t' + r['end_time_s'] + '\t' + 's' + '\n')
    elif format_type == 'csv_prob':
        with open(filename, 'w', ) as csvfile:
            fieldnames = ['start_time_s', 'end_time_s', 'music_prob', 'speech_prob']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in result:
                writer.writerow(r)


def main(audio_path, output_dir, format_type):
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
            export_result(result_csv_filename, file_result, format_type)

    else:
        file_result = smd.predict_audio(audio_path)
        result_csv_filename = os.path.join(output_dir, os.path.basename(audio_path) + '.csv')
        export_result(result_csv_filename, file_result, format_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TVSM Inference', description='TVSM Inference')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'csv_prob'])
    args = parser.parse_args()
    main(args.audio_path, args.output_dir, args.format)
