from collections import OrderedDict

import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self._net = nn.Sequential(OrderedDict([
            # 2048x4x4 -> 1024x8x8
            ('deconv1', nn.ConvTranspose2d(2048, 1024, 4, 2, 1)),
            ('relu1', nn.ReLU(inplace=True)),

            # 1024x8x8 -> 512x16x16
            ('deconv2', nn.ConvTranspose2d(1024, 512, 4, 2, 1)),
            ('relu2', nn.ReLU(inplace=True)),

            # 512x16x16 -> 256x32x32
            ('deconv3', nn.ConvTranspose2d(512, 256, 4, 2, 1)),
            ('relu3', nn.ReLU(inplace=True)),

            # 256x32x32 -> 128x64x64
            ('deconv4', nn.ConvTranspose2d(256, 128, 4, 2, 1)),
            ('relu4', nn.ReLU(inplace=True)),

            # 128x64x64 -> 3x128x128
            ('deconv5', nn.ConvTranspose2d(128, 3, 4, 2, 1)),
            ('tanh', nn.Tanh())
        ]))

    def forward(self, x):
        assert x.size()[-3:] == (2048, 4, 4)
        return self._net(x)


if __name__ == '__main__':
    import torch
    from encoder import Encoder

    enc = Encoder()
    dec = Decoder()

    x = torch.autograd.Variable(torch.rand(10, 3, 128, 128))
    z = enc.forward(x)
    x_ = dec.forward(z)

    print x_.data.cpu().numpy()
