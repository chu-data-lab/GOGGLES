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


is_cuda = torch.cuda.is_available()
put_on_gpu = lambda x: x.cuda() if is_cuda else x


def reproject(model, dataset):
    num_prototypes = model.num_prototypes

    all_patches = list()
    for image, label, attributes, _ in dataset:
        x = image.view((1,) + image.size())
        x = put_on_gpu(torch.autograd.Variable(x))
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


def load_datasets(input_image_size, *dataset_args):
    assert len(dataset_args) == 3

    transform_random_flip = transforms.RandomHorizontalFlip()
    transform_scale =  transforms.Scale((input_image_size, input_image_size))
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


def cub_dataset_batch_collate(batch):
    batch = zip(*batch)  # transpose

    image, label, attributes, \
    num_nonzero_attributes = batch

    image = torch.stack(image)
    label = torch.LongTensor(label)
    attributes = torch.stack([torch.LongTensor(a) for a in attributes])
    padding_idx = torch.LongTensor(num_nonzero_attributes)

    return image, label, attributes, padding_idx


def save_tensor_as_image(x, filepath):
    image = x.data.cpu().numpy()[0]
    image = 255. * (0.5 + (image / 2.))
    image = np.array(image, dtype=np.uint8)
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    image = Image.fromarray(image)
    image.save(filepath)


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
        collate_fn=cub_dataset_batch_collate,
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

            epoch_loss += loss.data[0]

        pbar.set_description('average_epoch_loss=%0.05f' % (epoch_loss / i))

        if (epoch % 5 == 0) or (epoch == num_epochs):
            reproject(model, train_dataset_deterministic)

        for i, (image, label, attributes, _) in enumerate(test_dataset):
            x = image.view((1,) + image.size())
            x = put_on_gpu(torch.autograd.Variable(x))
            z, z_patches, reconstructed_x = model(x)

            save_tensor_as_image(
                reconstructed_x, os.path.join(OUT_DIR, 'images', '%d-%d.png' % (epoch, i)))

    torch.save(model, os.path.join(MODEL_DIR, 'model.pt'))


if __name__ == '__main__':
    main()
