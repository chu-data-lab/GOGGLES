from collections import OrderedDict

import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, num_in_channels, num_conv_volumes=3,
                 kernel_size=4, conv_stride=2, conv_padding=1):
        super(Decoder, self).__init__()

        self._num_in_channels = num_in_channels
        self._num_conv_volumes = num_conv_volumes

        layers = list()
        channel_shrinkage_factor = 2
        in_channels = num_in_channels
        out_channels = num_in_channels / channel_shrinkage_factor
        conv_spec = (kernel_size, conv_stride, conv_padding)
        for i in range(num_conv_volumes - 1):
            layers.append(('deconv%d' % (i + 1), nn.ConvTranspose2d(in_channels, out_channels, *conv_spec),))
            layers.append(('relu%d' % (i + 1), nn.ReLU(inplace=True),))

            in_channels = out_channels
            out_channels /= channel_shrinkage_factor

        i = num_conv_volumes
        layers.append(('deconv%d' % i, nn.ConvTranspose2d(in_channels, 3, *conv_spec),))
        layers.append(('tanh', nn.Tanh()))

        self._net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        assert x.size()[1] == self._num_in_channels
        return self._net(x)


if __name__ == '__main__':
    import torch
    from encoder import Encoder

    input_image_size = 64
    expected_image_shape = (3, input_image_size, input_image_size)

    enc = Encoder(input_image_size)
    dec = Decoder(enc.num_out_channels)

    x = torch.autograd.Variable(torch.rand(1, *expected_image_shape))
    z = enc.forward(x)
    x_ = dec.forward(z)

    print(dec)
    print(x_.size())
    assert x_.size()[-3:] == expected_image_shape
