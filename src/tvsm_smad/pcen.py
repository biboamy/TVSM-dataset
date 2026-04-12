"""PCEN (Per-Channel Energy Normalization) transform for audio preprocessing."""

import numpy as np
import torch
import torch.nn as nn


class F2M(nn.Module):
    """Convert STFT to Mel-frequency spectrogram using triangular filter banks.

    Args:
        n_mels: Number of Mel bins.
        sr: Sample rate.
        f_max: Maximum frequency (default: sr // 2).
        f_min: Minimum frequency.
        n_fft: FFT size.
        onesided: Whether the STFT is one-sided.
    """

    def __init__(self, n_mels=128, sr=16000, f_max=None, f_min=0.0, n_fft=1024, onesided=True):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        if onesided:
            self.n_fft = self.n_fft // 2 + 1
        self._init_buffers()

    def _init_buffers(self):
        m_min = 0.0 if self.f_min == 0 else 2595 * np.log10(1.0 + (self.f_min / 700))
        m_max = 2595 * np.log10(1.0 + (self.f_max / 700))

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = 700 * (10 ** (m_pts / 2595) - 1)

        bins = torch.floor(((self.n_fft - 1) * 2) * f_pts / self.sr).long()

        fb = torch.zeros(self.n_fft, self.n_mels)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[f_m_minus:f_m, m - 1] = (
                    (torch.arange(f_m_minus, f_m) - f_m_minus).float() / (f_m - f_m_minus)
                )
            if f_m != f_m_plus:
                fb[f_m:f_m_plus, m - 1] = torch.div(
                    (float(f_m_plus) - torch.arange(f_m, f_m_plus)), (f_m_plus - f_m)
                )
        self.register_buffer("fb", fb)

    def forward(self, spec_f):
        return torch.matmul(spec_f, self.fb)


def pcen(x, eps=1e-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):
    """Per-Channel Energy Normalization."""
    frames = x.split(1, -2)
    m_frames = []
    last_state = None
    for frame in frames:
        if last_state is None:
            last_state = frame
            m_frames.append(frame)
            continue
        m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta**r
    return pcen_


class PCENTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.f2m = F2M(n_fft=1024, n_mels=128)

    def forward(self, x, is_mel=True):
        if not is_mel:
            x = torch.stft(x, n_fft=1024, hop_length=512, return_complex=True).abs()
            x = self.f2m(x.permute(0, 2, 1))
        x = pcen(x, eps=1e-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False)
        return x
