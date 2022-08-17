import torch.nn as nn
from torch.nn import LSTM, Linear, BatchNorm1d
import torch

class RNN(nn.Module):
    def __init__(self, inp_feature, oup_feature, rnn_layers=1, hidden_size=64):
        super().__init__()

        self.fc1 = Linear(inp_feature, hidden_size, bias=False)
        self.bn1 = BatchNorm1d(hidden_size)

        lstm_hidden_size = hidden_size // 2
    
        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        self.fc2 = Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)
        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(in_features=hidden_size, out_features=oup_feature, bias=False)

    def forward(self, x, mel=None):
        '''
        Input: (nb_samples, nb_frames, nb_timesteps)
        Output:(nb_samples, nb_frames, nb_timesteps)
        '''
        x = self.fc1(x.permute(0, 2, 1))
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out = self.lstm(x)
        layer = torch.cat([x, lstm_out[0]], -1)
        x = self.fc2(layer)
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.fc3(x)

        return x.permute(0, 2, 1), layer
