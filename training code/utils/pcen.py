import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../')
from .f2m import F2M

# given a melspectrogram, calculate the PCEN
def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):
    '''
    Input: (n_samples, n_frames, n_bins)
    Output: (n_samples, n_frames, n_bins)
    '''
    frames = x.split(1, -2)
    m_frames = []
    last_state = None
    for frame in frames:
        if last_state is None:
            last_state = frame
            m_frames.append(frame)
            continue
        if training:
            m_frame = ((1 - s) * last_state) + (s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r

    return pcen_

# calculate the PCEN
class PCENTransform(nn.Module):
    def __init__(self, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=False, 
            use_cuda_kernel=False, **stft_kwargs):
        super().__init__()
        self.use_cuda_kernel = use_cuda_kernel
        if trainable:
            self.s = nn.Parameter(torch.Tensor([s]))
            self.alpha = nn.Parameter(torch.Tensor([alpha]))
            self.delta = nn.Parameter(torch.Tensor([delta]))
            self.r = nn.Parameter(torch.Tensor([r]))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable
        self.stft_kwargs = stft_kwargs
        mel_keys = {"n_mels", "sr", "f_max", "f_min", "n_fft"}
        mel_keys = set(stft_kwargs.keys()).intersection(mel_keys)
        mel_kwargs = {k: stft_kwargs[k] for k in mel_keys}
        stft_keys = set(stft_kwargs.keys()) - mel_keys
        self.n_fft = stft_kwargs["n_fft"]
        self.stft_kwargs = {k: stft_kwargs[k] for k in stft_keys}
        self.f2m = F2M(**mel_kwargs)

    def forward(self, x, isMel=True):
        '''
        Input: (n_samples, n_frames, n_bins) if isMel = True / (n_samples, n_frames) if isMel = False
        Output: (n_samples, n_frames, n_bins)
        '''
        if not isMel:
            x = torch.stft(x, self.n_fft, **self.stft_kwargs).norm(dim=-1, p=2)
            x = self.f2m(x.permute(0, 2, 1))
        x = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable)
        return x
