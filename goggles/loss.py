import torch
import torch.nn as nn
import torch.nn.functional as F

from goggles.utils.functional import (pairwise_cosine_similarities,
                                      pairwise_squared_euclidean_distances)


def my_loss_function(reconstructed_x, z_patches, prototypes, padding_idx, x, lambda_=0.01):
    """
        reconstructed_x : batch_size, channels, height, width
        z_patches       : batch_size, num_patches, embedding_dim
        prototypes      : batch_size, num_prototypes, embedding_dim
        padding_idx     : batch_size
        x               : batch_size, channels, height, width
    """
    assert not x.requires_grad

    batch_size = x.size(0)

    loss = F.mse_loss(reconstructed_x, x, size_average=False)
    for i in range(batch_size):
        image_patches = z_patches[i]
        image_prototypes = prototypes[i][:padding_idx[i]]

        dists = pairwise_squared_euclidean_distances(
            image_prototypes, image_patches)
        min_dists = torch.min(dists, dim=1)[0]

        prototype_loss = torch.sum(min_dists)
        loss = loss + (lambda_ * prototype_loss)

    loss = loss / batch_size
    return loss


class CustomLoss1(object):
    def __init__(self, lambda_val=0.01):
        super(CustomLoss1, self).__init__()

        self._lambda_val = lambda_val
        self._reconstruction_loss = F.mse_loss
        self._xentropy_loss = nn.CrossEntropyLoss()

    def __call__(self, reconstructed_x, z_patches, prototypes, padding_idx, x):
        """
            reconstructed_x : batch_size, channels, height, width
            z_patches       : batch_size, num_patches, embedding_dim
            prototypes      : batch_size, num_prototypes, embedding_dim
            padding_idx     : batch_size
            x               : batch_size, channels, height, width
        """
        assert not x.requires_grad

        lambda_val = self._lambda_val
        batch_size = x.size(0)

        loss = lambda_val * self._reconstruction_loss(
            reconstructed_x, x, size_average=False)

        for i in range(batch_size):
            image_patches = z_patches[i]
            associated_prototypes = prototypes[i][:padding_idx[i]]

            dists = pairwise_squared_euclidean_distances(
                associated_prototypes, image_patches)
            nearest_patch_idxs = torch.min(dists, dim=1)[1]

            loss += self._xentropy_loss(-dists, nearest_patch_idxs)

        loss = loss / batch_size
        return loss

    def cuda(self, device=None):
        self._xentropy_loss = \
            self._xentropy_loss.cuda(device)


class CustomLoss2(object):
    def __init__(self, lambda_val=0.01):
        super(CustomLoss2, self).__init__()

        self._is_cuda = False

        self._lambda_val = lambda_val
        self._reconstruction_loss = F.mse_loss
        self._bxent_loss = nn.BCEWithLogitsLoss(size_average=False)

    def __call__(self, reconstructed_x, z_patches,
                 prototypes, padding_idx, x,
                 negative_prototypes=None):
        """
            reconstructed_x : batch_size, channels, height, width
            z_patches       : batch_size, num_patches, embedding_dim
            prototypes      : batch_size, num_prototypes, embedding_dim
            padding_idx     : batch_size
            x               : batch_size, channels, height, width
        """
        assert not x.requires_grad

        lambda_val = self._lambda_val
        batch_size = x.size(0)
        num_patches = z_patches.size(1)

        loss = self._reconstruction_loss(
            reconstructed_x, x, size_average=False)

        for i in range(batch_size):
            image_patches = z_patches[i]
            associated_prototypes = prototypes[i][:padding_idx[i]]

            sims = pairwise_cosine_similarities(  # num_prototypes x num_patches
                associated_prototypes, image_patches)
            nearest_patch_idxs = torch.max(sims, dim=1)[1]

            targets = self._make_cuda(torch.eye(num_patches))
            targets = targets[nearest_patch_idxs]

            loss += lambda_val * self.custom_cross_entropy(sims, targets)

            if negative_prototypes is not None:
                not_associated_prototypes = negative_prototypes[i]

                negative_sims = pairwise_cosine_similarities(
                    not_associated_prototypes, image_patches)
                negative_targets = self._make_cuda(
                    torch.zeros_like(negative_sims))

                loss += lambda_val * self.custom_cross_entropy(
                    negative_sims, negative_targets)

        loss = loss / batch_size
        return loss

    def _make_cuda(self, x):
        return x.cuda() if self._is_cuda else x

    def cuda(self, device=None):
        self._is_cuda = True

        self._bxent_loss = \
            self._bxent_loss.cuda(device)

        return self

    @staticmethod
    def custom_cross_entropy(x, y):
        sigmoid_x = torch.sigmoid(x)
        sigmoid_x2 = torch.sigmoid(x ** 2)
        neg_log_sigmoid_x = -1 * torch.log(sigmoid_x)
        neg_log_1_minus_sigmoid_x2 = -1 * torch.log(1 - sigmoid_x2)

        l1 = torch.mul(y, neg_log_sigmoid_x)
        l2 = torch.mul(1 - y, neg_log_1_minus_sigmoid_x2)

        return torch.sum(l1 + l2)


if __name__ == '__main__':
    reconstructed_x = torch.autograd.Variable(torch.rand(4, 3, 64, 64))
    z_patches = torch.autograd.Variable(torch.rand(4, 64, 512))
    attribute_prototypes = torch.autograd.Variable(torch.rand(4, 37, 512))
    padding_idx = torch.IntTensor([4, 5, 9, 37])
    x = torch.autograd.Variable(torch.rand(4, 3, 64, 64))

    print(my_loss_function(reconstructed_x, z_patches, attribute_prototypes, padding_idx, x))

    # attr = torch.IntTensor([
    #     [1, 2, 3, 4, 0, 0, 0],
    #     [1, 2, 0, 0, 0, 0, 0],
    #     [1, 2, 3, 4, 5, 6, 7]
    # ])
    #
    # print(attr.nonzero())
