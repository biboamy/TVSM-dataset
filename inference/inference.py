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
import json
import logging


def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


setup_logging()
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_SIZE = 512
N_FEATURES = 128
DURATION = 20

MUSIC_THRESHOLD = 0.5
SPEECH_THRESHOLD = 0.5

MIN_SEGMENT_LENGTH_S = 2
MAX_SEGMENT_MERGE_LENGTH_S = 2

here = os.path.dirname(os.path.abspath(__file__))
pseudo_model_path = os.path.join(here, 'models', 'TVSM-pseudo', 'epoch=28-step=67192.ckpt.torch.pt')


def model_creator():
    model = CRNN.CRNN()
    return model


class SMDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = model_creator()
        self.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.pcen_transform = T.Compose([
            torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_SIZE).to(self.device),
            PCENTransform().to(self.device)
        ])
        logger.info(f'Finish loading SMDetector device: {self.device}')

    def load_from_checkpoint(self, model_path):
        logger.info(f'Loading model from {model_path}')
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)

    def predict_audio(self, audio_path):
        # force load audio to SAMPLE_RATE and mono
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        y = np.expand_dims(y, 0)
        audio = torch.from_numpy(y).float()
        audio = audio.to(self.device)

        # librosa internally performs resampling and forces mono, there's no need to check for these again.

        audio_pcen_data = self.pcen_transform(audio)
        c_size = int(SAMPLE_RATE / HOP_SIZE * DURATION)
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
        frame_time = 1 / ((SAMPLE_RATE / HOP_SIZE) / 6)
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


def simple_merge_segments(segments):
    if not segments:
        return []
    merged_segments = [segments[0]]
    for i in range(1, len(segments)):
        prev = merged_segments[-1]
        current = segments[i]

        if prev['label'] == current['label']:
            merged_segments[-1]['end_time_s'] = current['end_time_s']
        else:
            merged_segments.append(current)

    # delete short segments
    merged_segments = [s for s in merged_segments if s['end_time_s'] - s['start_time_s'] >= MIN_SEGMENT_LENGTH_S]

    # merge adjacent segments
    filtered_merged_segments = [merged_segments[0]]
    for i in range(1, len(merged_segments)):
        prev_f = filtered_merged_segments[-1]
        current_f = merged_segments[i]
        if prev_f['label'] == current_f['label']:
            if current_f['start_time_s'] - prev_f['end_time_s'] <= MAX_SEGMENT_MERGE_LENGTH_S:
                filtered_merged_segments[-1]['end_time_s'] = current_f['end_time_s']
        else:
            filtered_merged_segments.append(current_f)

    return filtered_merged_segments


def write_csv(output_filename, fieldnames, result):
    with open(output_filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in result:
            writer.writerow(r)


def export_result(output_filename, result, format_type='csv'):
    if format_type == 'csv':
        fields_names = ['start_time_s', 'end_time_s', 'label']
        write_csv(output_filename, fields_names, result)
    elif format_type == 'csv_prob':
        fields_names = ['start_time_s', 'end_time_s', 'music_prob', 'speech_prob']
        write_csv(output_filename, fields_names, result)
    elif format_type == 'json':
        with open(output_filename, 'w') as jsonfile:
            json.dump(result, jsonfile)
    elif format_type == 'json_prob':
        with open(output_filename, 'w') as jsonfile:
            json.dump(result, jsonfile)
    else:
        logger.error(f'Invalid format type: {format_type}', )


def classify_label(music_prob, speech_prob):
    # more inclined towards finding all the music.
    if music_prob > MUSIC_THRESHOLD:
        return 'music'
    else:
        if speech_prob > SPEECH_THRESHOLD:
            return 'speech'
        else:
            return 'other'


def process_file(smd, audio_path, output_filename, format_type):
    try:
        audio_result = smd.predict_audio(audio_path)
        if format_type in ['csv', 'json']:
            labeled_segments = []
            for segment in audio_result:
                label = classify_label(segment['music_prob'], segment['speech_prob'])
                labeled_segments.append({
                    'start_time_s': round(float(segment['start_time_s']), 2),
                    'end_time_s': round(float(segment['end_time_s']), 2),
                    'label': label
                })
            result = simple_merge_segments(labeled_segments)
        else:
            result = audio_result
        export_result(output_filename, result, format_type)
        logger.info(f'Processed: {output_filename}', )

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {str(e)}")


def get_suffix(format_type):
    suffix = '.csv'

    if format_type == 'json':
        suffix = '.json'

    if format_type == 'json_prob':
        suffix = '.prob.json'

    if format_type == 'csv_prob':
        suffix = '.prob.csv'
    return suffix


def main(audio_path, output_dir, format_type):
    if not os.path.exists(audio_path):
        logger.error(f'No such file or directory: {audio_path}', )
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    smd = SMDetector(pseudo_model_path)

    suffix = get_suffix(format_type)

    if os.path.isdir(audio_path):
        all_files = []
        for root, dirs, files in os.walk(audio_path):
            for file in files:
                full_path = os.path.join(root, file)
                all_files.append(full_path)

        for full_path in tqdm.tqdm(all_files):
            result_csv_filename = os.path.join(output_dir, os.path.basename(full_path) + suffix)
            process_file(smd, full_path, result_csv_filename, format_type)

    else:
        result_csv_filename = os.path.join(output_dir, os.path.basename(audio_path) + suffix)
        process_file(smd, audio_path, result_csv_filename, format_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TVSM Inference', description='TVSM Inference')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'csv_prob', 'json', 'json_prob'])
    args = parser.parse_args()
    main(args.audio_path, args.output_dir, args.format)
