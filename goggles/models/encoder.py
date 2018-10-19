from collections import OrderedDict

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        nc_input = 3
        nc_filters = 128

        self._net = nn.Sequential(OrderedDict([
            # 3x128x128 -> 128x64x64
            ('conv1', nn.Conv2d(nc_input, nc_filters, 4, 2, 1)),
            ('relu1', nn.ReLU(inplace=True)),

            # 128x64x64 -> 256x32x32
            ('conv2', nn.Conv2d(nc_filters, nc_filters * 2, 4, 2, 1)),
            ('relu2', nn.ReLU(inplace=True)),

            # 256x32x32 -> 512x16x16
            ('conv3', nn.Conv2d(nc_filters * 2, nc_filters * 4, 4, 2, 1)),
            ('relu3', nn.ReLU(inplace=True)),

            # 512x16x16 -> 1024x8x8
            ('conv4', nn.Conv2d(nc_filters * 4, nc_filters * 8, 4, 2, 1)),
            ('relu4', nn.ReLU(inplace=True)),

            # 1024x8x8 -> 2048x4x4
            ('conv5', nn.Conv2d(nc_filters * 8, nc_filters * 16, 4, 2, 1)),
            ('relu5', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        assert x.size()[-3:] == (3, 128, 128)
        return self._net(x)


if __name__ == '__main__':
    import torch

    expected_image_shape = (3, 128, 128)
    input_tensor = torch.autograd.Variable(torch.rand(1, *expected_image_shape))

    net = Encoder()
    output_tensor = net(input_tensor)
    print output_tensor.size()
