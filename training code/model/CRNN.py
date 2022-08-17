import torch.nn as nn
from torch.nn import GRU, Linear, BatchNorm2d, BatchNorm1d, Conv2d, ReLU, MaxPool2d
import torch

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Sequential(
            Conv2d(1, 64, 3, 1, 1),
            ReLU(),
            MaxPool2d((2, 1)),
            BatchNorm2d(64)
        )

        self.c2 = nn.Sequential(
            Conv2d(64, 64, 11, 1, 5),
            ReLU(),
            MaxPool2d((2, 1)),
            BatchNorm2d(64)
        )

        self.c3 = nn.Sequential(
            Conv2d(64, 16, 11, 1, 5),
            ReLU(),
            MaxPool2d((2, 1)),
            BatchNorm2d(16)
        )

        self.lstm1 = nn.Sequential(
            GRU(input_size=256, hidden_size=80, num_layers=1, bidirectional=True, batch_first=True ),
        )
        self.b1 = BatchNorm1d(160)
        self.lstm2 = nn.Sequential(
            GRU(input_size=160, hidden_size=40, num_layers=1, bidirectional=True, batch_first=True )
        )
        self.b2 = BatchNorm1d(80)

        self.last = Linear(80, 2)

    def forward(self, x, mel=None):
        '''
        Input: (nb_samples, nb_frames, nb_timesteps)
        Output:(nb_samples, nb_frames, nb_timesteps)
        '''
        x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.b1(self.lstm1(x.reshape(x.shape[0], -1, x.shape[-1]).permute(0, 2, 1))[0].permute(0, 2, 1))
        x = self.b2(self.lstm2(x.permute(0, 2, 1))[0].permute(0, 2, 1))

        x = self.last(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)
