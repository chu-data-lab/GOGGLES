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


def main():
    input_image_size = 128
    batch_size = 64
    filter_species_ids = [14, 90]
    patch_size = 1
    num_epochs = 5000

    train_dataset_with_random_transformation, train_dataset_with_non_random_transformation, \
        test_dataset = CUBDataset.load_dataset_splits(CUB_DATA_DIR, input_image_size, filter_species_ids)

    train_dataloader = DataLoader(
        train_dataset_with_random_transformation,
        collate_fn=CUBDataset.custom_collate_fn,
        batch_size=batch_size, shuffle=True)

    model = _make_cuda(SemanticAutoencoder(
        input_image_size, patch_size, train_dataset_with_random_transformation.num_attributes))

    criterion = _make_cuda(CustomLoss2())
    optimizer = optim.Adam(ifilter(lambda p: p.requires_grad, model.parameters()))

    pbar = tqdm(range(1, num_epochs + 1))
    for epoch in pbar:
        epoch_loss = 0.

        for i, (image, label, attribute_labels, padding_idx) in enumerate(train_dataloader, 1):
            model.zero_grad()

            x = _make_cuda(torch.autograd.Variable(image))
            z, z_patches, reconstructed_x = model(x)

            prototype_labels = _make_cuda(attribute_labels)
            prototypes = model.prototypes(prototype_labels)

            loss = criterion(reconstructed_x, z_patches, prototypes, padding_idx, x)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        pbar.set_description('average_epoch_loss=%0.05f' % (epoch_loss / i))

        if (epoch % 5 == 0) or (epoch == num_epochs):
            nearest_patches_for_prototypes = \
                model.get_nearest_patches_for_prototypes(
                    train_dataset_with_non_random_transformation)

            model.reproject_prototypes(nearest_patches_for_prototypes)

            if (epoch % 100 == 0) or (epoch == num_epochs):
                save_prototype_patch_visualization(
                    model, train_dataset_with_non_random_transformation,
                    nearest_patches_for_prototypes, os.path.join(OUT_DIR, 'prototypes'))

                for i_, (image, image_label, attribute_labels, _) in enumerate(test_dataset):
                    x = image.view((1,) + image.size())
                    x = _make_cuda(torch.autograd.Variable(x))
                    z, z_patches, reconstructed_x = model(x)

                    reconstructed_image = get_image_from_tensor(reconstructed_x)
                    reconstructed_image.save(os.path.join(OUT_DIR, 'images', '%d-%d.png' % (epoch, i_)))

                model.save_weights(os.path.join(MODEL_DIR, 'model.pt'))
    model.save_weights(os.path.join(MODEL_DIR, 'model.pt'))


if __name__ == '__main__':
    main()
