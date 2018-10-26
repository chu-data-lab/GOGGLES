import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from patch import Patch
from prototype import Prototypes


class SemanticAutoencoder(nn.Module):
    def __init__(self, input_size, encoded_patch_size, num_prototypes):
        super(SemanticAutoencoder, self).__init__()

        self.num_prototypes = num_prototypes

        self._encoder_net = Encoder(input_size)
        self._decoder_net = Decoder(self._encoder_net.num_out_channels)

        encoded_output_size = self._encoder_net.output_size
        assert encoded_patch_size < encoded_output_size
        self._patches = Patch.from_spec(
            (encoded_output_size, encoded_output_size),
            (encoded_patch_size, encoded_patch_size))

        dim_prototypes = self._encoder_net.num_out_channels * (encoded_patch_size ** 2)
        self.prototypes = Prototypes(num_prototypes + 1, dim_prototypes, padding_idx=0)
        self.prototypes.weight.requires_grad = False  # freeze embeddings

    def forward(self, x):
        z = self._encoder_net(x)
        reconstructed_x = self._decoder_net(z)

        z_patches = [patch(z) for patch in self._patches]  # [patch1 (Tensor{batch_size, dim}), patch2, ..]
        z_patches = torch.stack(z_patches)  # num_patches, batch_size, embedding_dim
        z_patches = z_patches.transpose(0, 1)  # batch_size, num_patches, embedding_dim

        return z, z_patches, reconstructed_x


if __name__ == '__main__':
    from itertools import ifilter
    input_image_size = 64
    expected_image_shape = (3, input_image_size, input_image_size)
    input_tensor = torch.autograd.Variable(torch.rand(5, *expected_image_shape))

    net = SemanticAutoencoder(input_image_size, 1, 10)
    print net
    for p in ifilter(lambda p: p.requires_grad, net.parameters()):
        print p.size()
    # print
    # z, z_patches, reconstructed_x = net(input_tensor)
    # print z.size()
    # print reconstructed_x.size()
    # print z_patches[0].size()
    # print len(z_patches)

    print net.prototypes.weight[1]
