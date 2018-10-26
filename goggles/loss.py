import torch
import torch.nn as nn
import torch.nn.functional as F


def my_loss_function(reconstructed_x, z_patches, attribute_prototypes, padding_idx, x, lambda_=0.01):
    """
        reconstructed_x      : batch_size, channels, height, width
        z_patches            : batch_size, num_patches, embedding_dim
        attribute_prototypes : batch_size, num_prototypes, embedding_dim
        padding_idx          : batch_size
        x                    : batch_size, channels, height, width
    """
    assert not x.requires_grad  # d(Loss)/dw or d(w1)/d(w2) in computational graph

    batch_size = x.size(0)
    num_patches = z_patches.size(1)
    num_prototypes = attribute_prototypes.size(1)
    embedding_dim = attribute_prototypes.size(2)

    attribute_prototypes = attribute_prototypes.unsqueeze(-2)
    attribute_prototypes = attribute_prototypes.expand(batch_size, num_prototypes, num_patches, embedding_dim)

    loss = F.mse_loss(reconstructed_x, x, size_average=False)

    pdist = nn.PairwiseDistance(p=2)
    for i in range(batch_size):
        num_attributes_in_image = padding_idx[i]
        for j in range(num_attributes_in_image):
            prototype = attribute_prototypes[i][j]
            image_patches = z_patches[i]

            distances = pdist(prototype, image_patches)
            min_dist = torch.min(distances)

            loss = loss + (lambda_ * min_dist)

    loss = loss / batch_size
    return loss


if __name__ == '__main__':
    reconstructed_x = torch.autograd.Variable(torch.rand(4, 3, 64, 64))
    z_patches = torch.autograd.Variable(torch.rand(4, 64, 512))
    attribute_prototypes = torch.autograd.Variable(torch.rand(4, 37, 512))
    padding_idx = torch.IntTensor([4, 5, 9, 37])
    x = torch.autograd.Variable(torch.rand(4, 3, 64, 64))

    print my_loss_function(reconstructed_x, z_patches, attribute_prototypes, padding_idx, x)

    # attr = torch.IntTensor([
    #     [1, 2, 3, 4, 0, 0, 0],
    #     [1, 2, 0, 0, 0, 0, 0],
    #     [1, 2, 3, 4, 5, 6, 7]
    # ])
    #
    # print attr.nonzero()
