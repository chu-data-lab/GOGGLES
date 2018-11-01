try:
    from itertools import ifilter
except ImportError:
    ifilter = filter
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

from constants import *
from data.cub.dataset import CUBDataset
from loss import my_loss_function
from models.semantic_ae import SemanticAutoencoder
from utils.vis import get_image_from_tensor


is_cuda = torch.cuda.is_available()
put_on_gpu = lambda x: x.cuda() if is_cuda else x


def load_datasets(input_image_size, *dataset_args):
    assert len(dataset_args) == 3

    transform_random_flip = transforms.RandomHorizontalFlip()
    transform_scale = transforms.Scale((input_image_size, input_image_size))
    transform_to_tensor = transforms.ToTensor()
    transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    random_transformation = transforms.Compose([
        transform_random_flip, transform_scale, transform_to_tensor, transform_normalize])
    deterministic_transformation = transforms.Compose([
        transform_scale, transform_to_tensor, transform_normalize])

    train_dataset_random = CUBDataset(
        *dataset_args, transform=random_transformation, is_training=True)

    train_dataset_deterministic = CUBDataset(
        *dataset_args, transform=deterministic_transformation, is_training=True)

    test_dataset = CUBDataset(
        *dataset_args, transform=deterministic_transformation, is_training=False)

    return train_dataset_random, train_dataset_deterministic, test_dataset


def main():
    input_image_size = 64
    species1_id = 14
    species2_id = 90
    patch_size = 1
    num_epochs = 100

    train_dataset_random, train_dataset_deterministic, \
        test_dataset = load_datasets(input_image_size, CUB_DATA_DIR, species1_id, species2_id)

    train_dataloader = DataLoader(
        train_dataset_random,
        collate_fn=CUBDataset.custom_collate_fn,
        batch_size=4, shuffle=True)

    model = put_on_gpu(SemanticAutoencoder(
        input_image_size,
        patch_size,
        train_dataset_random.num_attributes))

    optimizer = optim.Adam(ifilter(lambda p: p.requires_grad, model.parameters()))

    pbar = tqdm(range(1, num_epochs + 1))
    for epoch in pbar:
        epoch_loss = 0.

        for i, (image, label, attributes, padding_idx) in enumerate(train_dataloader, 1):
            model.zero_grad()

            x = put_on_gpu(torch.autograd.Variable(image))
            z, z_patches, reconstructed_x = model(x)

            attributes = put_on_gpu(attributes)
            attribute_prototypes = model.prototypes(attributes)

            loss = my_loss_function(reconstructed_x, z_patches, attribute_prototypes, padding_idx, x)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        pbar.set_description('average_epoch_loss=%0.05f' % (epoch_loss / i))

        if (epoch % 5 == 0) or (epoch == num_epochs):
            model.reproject_prototypes_to_dataset(train_dataset_deterministic)

        for i, (image, label, attributes, _) in enumerate(test_dataset):
            x = image.view((1,) + image.size())
            x = put_on_gpu(torch.autograd.Variable(x))
            z, z_patches, reconstructed_x = model(x)

            reconstructed_image = get_image_from_tensor(reconstructed_x)
            reconstructed_image.save(os.path.join(OUT_DIR, 'images', '%d-%d.png' % (epoch, i)))

    torch.save(model, os.path.join(MODEL_DIR, 'model.pt'))


if __name__ == '__main__':
    main()
