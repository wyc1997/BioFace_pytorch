import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

from CNN import CNNEncoder, FullyConnectedRegressor, Decoder

if __name__ == "__main__":
    encoder = CNNEncoder([3, 32, 64, 128, 256, 512])
    test = torch.ones([2, 3, 64, 64])

    summary(encoder, [[3, 64, 64]])
    output, feat_dict = encoder(test)
    print(output.shape)

    for n in feat_dict:
        print(feat_dict[n].shape)

    decoder = Decoder([3, 32, 64, 128, 256, 512])
    print(decoder.channel_list)
    out = decoder(output, feat_dict)
    print(out.shape)

    fc = FullyConnectedRegressor(512, 4, 20)
    intermediate = fc(output)
    print(intermediate.shape)
