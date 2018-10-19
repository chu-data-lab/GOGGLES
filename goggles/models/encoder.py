from collections import OrderedDict

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, num_conv_volumes,
                 kernel_size=2, conv_stride=2, conv_padding=0):
        super(Encoder, self).__init__()

        self._input_size = input_size
        self._num_conv_volumes = num_conv_volumes

        layers = list()
        in_channels = 3
        num_filters = 128
        conv_spec = (kernel_size, conv_stride, conv_padding)
        for i in range(num_conv_volumes):
            layers.append(('conv%d' % (i + 1), nn.Conv2d(in_channels, num_filters, *conv_spec),))
            layers.append(('relu%d' % (i + 1), nn.ReLU(inplace=True),))

            in_channels = num_filters
            num_filters *= 2

        self._net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        r = self._input_size
        assert x.size()[-3:] == (3, r, r)
        return self._net(x)


if __name__ == '__main__':
    import torch

    input_image_size = 128
    expected_image_shape = (3, input_image_size, input_image_size)
    input_tensor = torch.autograd.Variable(torch.rand(1, *expected_image_shape))

    net = Encoder(input_image_size, 4, kernel_size=2, conv_padding=0)
    print net
    output_tensor = net(input_tensor)
    print output_tensor.size()
