import torch
from torch import nn
import torch.nn.functional as F


def create_sequential(in_channel, out_channel, upconv=False):
    if upconv:
        in_channel = in_channel + out_channel
    s = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )
    return s


class CNNEncoder(nn.Module):
    def __init__(self, channel_list):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.ModuleDict()
        self.channel_list = channel_list
        for i, _ in enumerate(channel_list):
            if i == len(channel_list) - 1:
                break
            self.conv_layers[str(channel_list[i+1])] = create_sequential(channel_list[i], channel_list[i+1])

    def forward(self, x):
        feat_dict = {}
        for n in self.conv_layers:
            x = self.conv_layers[n](x)
            feat_dict[n] = x
            if int(n) == self.channel_list[-1]:
                break
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        output = x
        return output, feat_dict


class FullyConnectedRegressor(nn.Module):
    def __init__(self, in_channel, input_dim, output_dim):
        super(FullyConnectedRegressor, self).__init__()
        self.in_channel = in_channel
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim*input_dim*in_channel, in_channel)
        self.bn1 = nn.BatchNorm1d(in_channel)
        self.fc2 = nn.Linear(in_channel, in_channel)
        self.bn2 = nn.BatchNorm1d(in_channel)
        self.fc3 = nn.Linear(in_channel, output_dim)

    def forward(self, x):
        x = x.view((-1, self.input_dim * self.input_dim * self.in_channel))
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channel_list):
        super(Decoder, self).__init__()
        self.channel_list = list(reversed(channel_list))
        self.conv_dict = nn.ModuleDict()
        self.output_layer = nn.Conv2d(self.channel_list[-2], 1, kernel_size=3, padding=1)
        for i, _ in enumerate(self.channel_list):
            if i == (len(self.channel_list) - 2):
                break
            self.conv_dict[str(self.channel_list[i+1])] = create_sequential(self.channel_list[i],
                                                                      self.channel_list[i+1], True)

    def forward(self, x, feat_dict):
        for n in self.conv_dict:
            x = F.upsample(x, scale_factor=2)
            x = torch.cat([feat_dict[n], x], dim=1)
            # print(x.shape)
            x = self.conv_dict[n](x)
        # final output
        output = self.output_layer(x)
        return output
