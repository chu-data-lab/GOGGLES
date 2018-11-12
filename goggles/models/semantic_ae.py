from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from goggles.models.encoder import Encoder
from goggles.models.decoder import Decoder
from goggles.models.patch import Patch
from goggles.models.prototype import Prototypes
from goggles.utils.functional import pairwise_squared_euclidean_distances


class SemanticAutoencoder(nn.Module):
    def __init__(self, input_size, encoded_patch_size, num_prototypes):
        super(SemanticAutoencoder, self).__init__()
        self._is_cuda = False

        self.input_size = input_size
        self.encoded_patch_size = encoded_patch_size
        self.num_prototypes = num_prototypes

        self._encoder_net = Encoder(input_size)
        self._decoder_net = Decoder(self._encoder_net.num_out_channels)

        encoded_output_size = self._encoder_net.output_size
        assert encoded_patch_size <= encoded_output_size
        self._patches = Patch.from_spec(
            (encoded_output_size, encoded_output_size),
            (encoded_patch_size, encoded_patch_size))

        # pre-compute receptive fields for patches
        self._receptive_fields_for_patches = dict()
        for i in range(len(self._patches)):
            _ = self.get_receptive_field_for_patch(i)

        dim_prototypes = self._encoder_net.num_out_channels * (encoded_patch_size ** 2)
        self.prototypes = Prototypes(num_prototypes + 1, dim_prototypes, padding_idx=0)
        self.prototypes.weight.requires_grad = False  # freeze embeddings

    def _make_cuda(self, x):
        return x.cuda() if self._is_cuda else x

    def forward(self, x):
        z = self._encoder_net(x)
        reconstructed_x = self._decoder_net(z)

        z_patches = [patch(z) for patch in self._patches]  # [patch1:Tensor(batch_size, dim), patch2, ...]
        z_patches = torch.stack(z_patches)  # num_patches, batch_size, embedding_dim
        z_patches = z_patches.transpose(0, 1)  # batch_size, num_patches, embedding_dim

        return z, z_patches, reconstructed_x

    def cuda(self, device_id=None):
        self._is_cuda = True
        return super(SemanticAutoencoder, self).cuda(device_id)

    def get_receptive_field_for_patch(self, patch_idx):
        if patch_idx not in self._receptive_fields_for_patches:
            self.zero_grad()

            image_size = self.input_size
            batch_shape = (1, 3, image_size, image_size)

            x = self._make_cuda(torch.autograd.Variable(
                torch.rand(*batch_shape), requires_grad=True))
            z = self._encoder_net.forward(x)
            z_patch = self._patches[patch_idx].forward(z)

            torch.sum(z_patch).backward()

            rf = x.grad.data.cpu().numpy()
            rf = rf[0, 0]
            rf = list(zip(*np.where(np.abs(rf) > 1e-6)))

            (i_nw, j_nw), (i_se, j_se) = rf[0], rf[-1]

            rf_w, rf_h = (j_se - j_nw + 1,
                          i_se - i_nw + 1)

            self._receptive_fields_for_patches[patch_idx] = \
                (i_nw, j_nw), (rf_w, rf_h)

            self.zero_grad()

        return self._receptive_fields_for_patches[patch_idx]

    def get_nearest_patches_for_prototypes(self, dataset):
        all_patches = list()
        candidate_patch_idxs_dict = defaultdict(list)
        nearest_patches_for_prototypes = dict()

        for i, (image, _, attributes, num_nonzero_attributes) in enumerate(dataset):
            x = image.view((1,) + image.size())
            x = self._make_cuda(torch.autograd.Variable(x))
            z, z_patches, reconstructed_x = self.forward(x)

            attributes = attributes[:num_nonzero_attributes]

            patches = z_patches[0]
            for j, patch in enumerate(patches):
                patch_id = (i, j)
                all_patches_idx = len(all_patches)  # where patch will be added in all_patches

                for prototype_label in attributes:
                    candidate_patch_idxs_dict[prototype_label].append(
                        (all_patches_idx, patch_id))

                all_patches.append(patch)
        all_patches = torch.stack(all_patches)

        for k in range(1, self.num_prototypes + 1):
            candidate_patch_idxs = torch.LongTensor(
                [all_patches_idx
                 for all_patches_idx, _
                 in candidate_patch_idxs_dict[k]])
            candidate_patches = all_patches[candidate_patch_idxs]

            prototype_label = self._make_cuda(torch.LongTensor([k]))
            prototype = self.prototypes(prototype_label)

            dists = pairwise_squared_euclidean_distances(
                prototype, candidate_patches)

            nearest_patch_idx = torch.min(dists, dim=1)[1].data.cpu().numpy()[0]
            nearest_patch = candidate_patches[nearest_patch_idx]
            _, nearest_patch_id = candidate_patch_idxs_dict[k][nearest_patch_idx]  # (image_idx, patch_idx)
            nearest_patches_for_prototypes[k] = (nearest_patch_id, nearest_patch)

        return nearest_patches_for_prototypes

    def reproject_prototypes(self, nearest_patches_for_prototypes):
        for k, (nearest_image_patch_idx, nearest_patch) \
                in nearest_patches_for_prototypes.items():
            self.prototypes.weight[k].data = nearest_patch


if __name__ == '__main__':
    import torchvision.transforms as transforms

    from goggles.constants import *
    from goggles.data.cub.dataset import CUBDataset

    input_image_size = 64
    expected_image_shape = (3, input_image_size, input_image_size)

    transform_random_flip = transforms.RandomHorizontalFlip()
    transform_resize = transforms.Scale((input_image_size, input_image_size))
    transform_to_tensor = transforms.ToTensor()
    transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    random_transformation = transforms.Compose([
        transform_random_flip, transform_resize, transform_to_tensor, transform_normalize])

    test_dataset = CUBDataset('/Users/nilakshdas/Dev/GOGGLES/data/CUB_200_2011', transform=random_transformation)

    input_tensor = torch.autograd.Variable(torch.rand(5, *expected_image_shape))

    net = SemanticAutoencoder(input_image_size, 1, 10)
    print(net.get_nearest_patches_for_prototypes(test_dataset))
    # print(net.state_dict())
    # for p in ifilter(lambda p: p.requires_grad, net.parameters()):
    #     print(p.size())
    # print()
    # z, z_patches, reconstructed_x = net(input_tensor)
    # print(z.size())
    # print(reconstructed_x.size())
    # print(z_patches[0].size())
    # print(len(z_patches))

    # print(net.prototypes.weight[1])

    print(net.get_receptive_field_for_patch(0))
