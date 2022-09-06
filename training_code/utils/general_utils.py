import yaml
import os
from pytorch_lightning.callbacks import Callback
import argparse
import random
import torch
import numpy as np
import torch.nn as nn

# turn "None" string into None
def _strings_handle_none(arg):
    if arg.lower() in ['null', 'none']:
        return None
    else:
        return str(arg)

# turn "True/False" string into True/False
def _bool_string_to_bool(arg):
    if str(arg).lower() == "false":
        return False
    if str(arg).lower() == 'true':
        return True

# parse yaml format into parameters
def yaml_to_parser(yaml_path):
    parser = argparse.ArgumentParser()
    hparams = yaml.safe_load(open(yaml_path))
    for k, val in hparams.items():
        if isinstance(val, str):
            argparse_kwargs = {'type': _strings_handle_none, 'default': val}
        elif isinstance(val, bool):
            argparse_kwargs = {'type': _bool_string_to_bool, 'default': val}
        else:
            argparse_kwargs = {'type': eval, 'default': val}

        parser.add_argument('--{}'.format(k.replace('_', '-')), **argparse_kwargs)
    return parser

# write model output into csv format for evaluation
def model_output_to_csv(data, frame_second, des_path, name):
    '''
    :param data: (time, 2)
    :param sr: sampling rate
    :param hop_size:
    :return:
    '''
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    oup_file = open(os.path.join(des_path, os.path.splitext(name)[0] + '.csv'), "w")
    for i, frame in enumerate(data):
        if frame[0] == 1:
            oup_file.write(str(frame_second * i) + '\t' + str(frame_second * (i + 1)) + '\t' + 'm' + '\n')
        if frame[1] == 1:
            oup_file.write(str(frame_second * i) + '\t' + str(frame_second * (i + 1)) + '\t' + 's' + '\n')
    oup_file.close()

# fix the random seed
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Convert input audio to raw wav and output numpy array via ffmpeg
def ffmpeg_load_audio(filename, sr=48000, channels=6, start=None, duration=None, normalize=False, in_type=np.int16, out_type=np.float64, ffmpeg_path=None):
    """
    Input:
        filename: path for the m4a file
        sr: sampleing rate (int)
        channels: audio's channel (int)
        start: starting second (e.g. '10': start from 10 second)
        duration: chunk duration (e.g. '10': have the 10 second chunk)
        in_type: input type when loading from ffmpeg
        out_type: output type to be converted
    """
    import subprocess as sp
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]
    if ffmpeg_path is None:
        ffmpeg_path = 'ffmpeg'
    command = [
        ffmpeg_path,
        '-i', filename,
        '-f', format_string,
        '-ar', str(sr)
    ]
    if start is not None:
        command.append('-ss')
        command.append(start)
    if duration is not None:
        command.append('-t')
        command.append(duration)
    command.append('-')
    p = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)

    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.frombuffer(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels))
    if audio.size == 0:
        return audio, sr

    if issubclass(out_type, np.floating):
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max
    return audio, sr

# median filter for pytorch version
class MedianPool1d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        super(MedianPool1d, self).__init__()

        self.k = kernel_size
        self.stride = stride
        self.padding = padding  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            iw = x.shape[-1]
            if iw % self.stride == 0:
                pw = max(self.k - self.stride, 0)
            else:
                pw = max(self.k - (iw % self.stride), 0)
            pl = pw // 2
            pr = pw - pl
            padding = (pl, pr)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = torch.nn.functional.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k, self.stride)
        x = x.contiguous().view(x.size()[:3] + (-1,)).median(dim=-1)[0]
        return x