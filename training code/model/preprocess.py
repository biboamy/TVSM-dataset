import torchaudio
import torch.nn as nn
import torch
import numpy as np

class MelSpec(nn.Module):
    def __init__(self, sr, n_fft, hop_size):
        super(MelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_size)

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_mel, nb_timesteps)
        """
        return self.transform(x)

class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """
        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f

class vgg_mel(nn.Module):
    def __init__(self):
        super(vgg_mel, self).__init__()
        fft_length = 2 ** int(np.ceil(np.log(400) / np.log(2.0)))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=fft_length,
            win_length=400,
            hop_length=160,
            n_mels=64,
            f_min=125,
            f_max=7500,
            window_fn=torch.hann_window,
            wkwargs={'periodic':True}
        )
        self.resample = torchaudio.transforms.Resample(22050, 16000)

    def forward(self, data):
        '''
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_mel, nb_timesteps)
        '''
      
        mel = self.mel(data)

        log_mel = torch.log(mel + 0.01)

        return log_mel