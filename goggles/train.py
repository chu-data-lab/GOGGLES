try:
    from itertools import ifilter
except ImportError:
    ifilter = filter
import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

from goggles.constants import *
from goggles.data.cub.dataset import CUBDataset
from goggles.loss import CustomLoss2
from goggles.models.semantic_ae import SemanticAutoencoder
from goggles.utils.vis import get_image_from_tensor, save_prototype_patch_visualization


_make_cuda = lambda x: x.cuda() if torch.cuda.is_available() else x


def load_datasets(input_image_size, *dataset_args):
    assert len(dataset_args) == 2

    transform_random_flip = transforms.RandomHorizontalFlip()
    transform_resize = transforms.Scale((input_image_size, input_image_size))
    transform_to_tensor = transforms.ToTensor()
    transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    random_transformation = transforms.Compose([
        transform_random_flip, transform_resize, transform_to_tensor, transform_normalize])
    deterministic_transformation = transforms.Compose([
        transform_resize, transform_to_tensor, transform_normalize])

    train_dataset_random = CUBDataset(
        *dataset_args,
        transform=random_transformation,
        is_training=True)

    train_dataset_deterministic = CUBDataset(
        *dataset_args,
        transform=deterministic_transformation,
        is_training=True)

    test_dataset = CUBDataset(
        *dataset_args,
        required_attributes=train_dataset_deterministic.attributes,
        transform=deterministic_transformation,
        is_training=False)

    return train_dataset_random, train_dataset_deterministic, test_dataset


def main():
    input_image_size = 128
    batch_size = 64
    filter_species_ids = [14, 90]
    patch_size = 1
    num_epochs = 5000

    train_dataset_random, train_dataset_deterministic, \
        test_dataset = load_datasets(input_image_size, CUB_DATA_DIR, filter_species_ids)

    train_dataloader = DataLoader(
        train_dataset_random,
        collate_fn=CUBDataset.custom_collate_fn,
        batch_size=batch_size, shuffle=True)

    model = _make_cuda(SemanticAutoencoder(
        input_image_size,
        patch_size,
        train_dataset_random.num_attributes))

    criterion = _make_cuda(CustomLoss2())
    optimizer = optim.Adam(ifilter(lambda p: p.requires_grad, model.parameters()))

    pbar = tqdm(range(1, num_epochs + 1))
    for epoch in pbar:
        epoch_loss = 0.

        for i, (image, label, attributes, padding_idx) in enumerate(train_dataloader, 1):
            model.zero_grad()

            x = _make_cuda(torch.autograd.Variable(image))
            z, z_patches, reconstructed_x = model(x)

            attributes = _make_cuda(attributes)
            attribute_prototypes = model.prototypes(attributes)

            loss = criterion(reconstructed_x, z_patches, attribute_prototypes, padding_idx, x)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        pbar.set_description('average_epoch_loss=%0.05f' % (epoch_loss / i))

        if (epoch % 5 == 0) or (epoch == num_epochs):
            nearest_patches_for_prototypes = \
                model.get_nearest_patches_for_prototypes(
                    train_dataset_deterministic)
            model.reproject_prototypes(nearest_patches_for_prototypes)

            if (epoch % 100 == 0) or (epoch == num_epochs):
                save_prototype_patch_visualization(
                    model, train_dataset_deterministic,
                    nearest_patches_for_prototypes,
                    os.path.join(OUT_DIR, 'prototypes'))

                for i, (image, label, attributes, _) in enumerate(test_dataset):
                    x = image.view((1,) + image.size())
                    x = _make_cuda(torch.autograd.Variable(x))
                    z, z_patches, reconstructed_x = model(x)

                    reconstructed_image = get_image_from_tensor(reconstructed_x)
                    reconstructed_image.save(os.path.join(OUT_DIR, 'images', '%d-%d.png' % (epoch, i)))

    torch.save(model, os.path.join(MODEL_DIR, 'model.pt'))


if __name__ == '__main__':
    main()
