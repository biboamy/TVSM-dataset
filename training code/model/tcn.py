import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, oup_channels, kernel_size, dilate):
        super().__init__()

        pad = dilate * (kernel_size -1) // 2

        self.conv1 = nn.Conv1d(in_channels, oup_channels, kernel_size, padding=pad, dilation=dilate) #, groups=oup_channels)
        self.norm1 = nn.BatchNorm1d(oup_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(oup_channels, oup_channels, kernel_size, padding=pad, dilation=dilate) #, groups=oup_channels)
        self.norm2 = nn.BatchNorm1d(oup_channels)
        self.relu2 = nn.ReLU()

        self.conv_res = nn.Conv1d(in_channels, oup_channels, 1)
        self.act = nn.ReLU()

    def forward(self, inp):
        '''
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_mel, nb_timesteps)
        '''
        hid = self.relu1(self.norm1(self.conv1(inp)))
        hid = self.relu2(self.norm2(self.conv2(hid)))
        res = self.conv_res(inp)
        oup = res + hid
        oup = self.act(oup)

        return oup, hid

class tcn(nn.Module):
    def __init__(self, n_features, n_output, n_class, kernal_size, n_stacks, n_blocks):
        super().__init__()
        self.n_features = n_features
        self.n_output = n_output

        in_channels = n_features

        kernal_size = [3, 5, 5]
        n_output = [32, 16, 32]
        n_stacks = [9, 5, 2]
        n_blocks = [3, 7, 2]

        layers_1 = []
        for i_stack in range(n_stacks[0]):
            for i_block in range(n_blocks[0]):
                layers_1.append(ConvBlock(in_channels, n_output[0], kernal_size[0], 2**i_block))
                in_channels = n_output[0]
        
        layers_2 = []
        for i_stack in range(n_stacks[1]):
            for i_block in range(n_blocks[1]):
                layers_2.append(ConvBlock(in_channels, n_output[1], kernal_size[1], 2**i_block))
                in_channels = n_output[1]
        
        layers_3 = []
        for i_stack in range(n_stacks[2]):
            for i_block in range(n_blocks[2]):
                layers_3.append(ConvBlock(in_channels, n_output[2], kernal_size[2], 2**i_block))
                in_channels = n_output[2]
        
        self.layers1 = nn.ModuleList(layers_1)
        self.layers2 = nn.ModuleList(layers_2)
        self.layers3 = nn.ModuleList(layers_3)
        self.linear = nn.Linear(n_output[2], n_class)

    def forward(self, inp):
        '''
        Input: (nb_samples, nb_features, nb_timesteps)
        Output:(nb_samples, nb_features, nb_timesteps)
        '''
        skip_sum1 = 0.0
        skip_sum2 = 0.0
        skip_sum3 = 0.0
        for layer in self.layers1:
            inp, skip = layer(inp)
            skip_sum1 = skip_sum1 + skip
        
        for layer in self.layers2:
            inp, skip = layer(inp)
            skip_sum2 = skip_sum2 + skip
        
        for layer in self.layers3:
            skip_sum2, skip = layer(skip_sum2)
            skip_sum3 = skip_sum3 + skip
        
        oup = self.linear(skip_sum3.permute(0, 2, 1))
        return oup.permute(0, 2, 1)