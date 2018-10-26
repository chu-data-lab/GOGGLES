try:
    from itertools import ifilter
except ImportError:
    ifilter = filter
import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from constants import *
from data.cub.dataset import CUBDataset
from loss import my_loss_function
from models.semantic_ae import SemanticAutoencoder


def reproject(model, dataset):
    num_prototypes = model.num_prototypes

    all_patches = list()
    for image, label, attributes, _ in dataset:
        x = image.view((1,) + image.size())
        x = torch.autograd.Variable(x)
        z, z_patches, reconstructed_x = model(x)

        patches = z_patches[0]
        for patch in patches:
            all_patches.append(patch.data.cpu().numpy())
    all_patches = np.array(all_patches)

    for j in range(1, num_prototypes + 1):
        prototype = model.prototypes.weight[j].data.cpu().numpy()

        dists = np.linalg.norm(prototype - all_patches, ord=2, axis=1)

        nearest_patch_idx = np.argmin(dists)
        nearest_patch = all_patches[nearest_patch_idx]

        model.prototypes.weight[j].data = torch.FloatTensor(nearest_patch)


def main():
    species1_id = 14
    species2_id = 90

    input_image_size = 64
    patch_size = 1

    train_dataset_random = CUBDataset(root=CUB_DATA_DIR,
                                      species1_id=species1_id, species2_id=species2_id,
                                      transform=transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Scale((input_image_size, input_image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                      is_training=True)

    train_dataset_deterministic = CUBDataset(root=CUB_DATA_DIR,
                                             species1_id=species1_id, species2_id=species2_id,
                                             transform=transforms.Compose([
                                                 transforms.Scale((input_image_size, input_image_size)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                             is_training=True)

    test_dataset = CUBDataset(root=CUB_DATA_DIR,
                              species1_id=species1_id, species2_id=species2_id,
                              transform=transforms.Compose([
                                  transforms.Scale((input_image_size, input_image_size)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                              is_training=False)

    train_shuffled_dataloader = torch.utils.data.DataLoader(train_dataset_random, batch_size=4, shuffle=True)

    model = SemanticAutoencoder(input_image_size, patch_size, train_dataset_random.num_attributes)
    optimizer = optim.Adam(ifilter(lambda p: p.requires_grad, model.parameters()))

    num_epochs = 100
    pbar = tqdm(range(1, num_epochs + 1))
    for epoch in pbar:
        total_loss = 0.
        for i, batch in enumerate(train_shuffled_dataloader, 1):
            model.zero_grad()

            x = torch.autograd.Variable(batch[0])
            z, z_patches, reconstructed_x = model(x)

            attributes = torch.stack(batch[2]).t()  # batch_size, num_attributes
            attribute_prototypes = model.prototypes(attributes)  # batch_size, num_attributes, embedding_dim
            padding_idx = batch[3]

            loss = my_loss_function(reconstructed_x, z_patches, attribute_prototypes, padding_idx, x)

            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]

        pbar.set_description('average_epoch_loss=%0.05f' % (total_loss / i))

        if (epoch % 5 == 0) or (epoch == 1):  # TODO: reset
            reproject(model, train_dataset_deterministic)

        for i, (image, label, attributes, _) in enumerate(test_dataset):
            x = image.view((1,) + image.size())
            x = torch.autograd.Variable(x)
            z, z_patches, reconstructed_x = model(x)

            reconstructed_image = reconstructed_x.data.cpu().numpy()[0]
            reconstructed_image = 255. * (0.5 + (reconstructed_image / 2.))
            reconstructed_image = np.array(reconstructed_image, dtype=np.uint8)
            reconstructed_image = np.swapaxes(reconstructed_image, 0, 1)
            reconstructed_image = np.swapaxes(reconstructed_image, 1, 2)
            reconstructed_image = Image.fromarray(reconstructed_image)
            reconstructed_image.save(os.path.join(OUT_DIR, 'images', '%d-%d.png' % (epoch, i)))

    model.save_state_dict(os.path.join(MODEL_DIR, 'model.pt'))


if __name__ == '__main__':
    main()
