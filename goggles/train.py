try:
    from itertools import ifilter
except ImportError:
    ifilter = filter
import os
import time

from tensorboardX import SummaryWriter
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

    writer = SummaryWriter(os.path.join(LOG_DIR, time.strftime('%Y%m%d-%H%M%S')))
    writer.add_scalar('param/input_image_size', input_image_size)
    writer.add_scalar('param/batch_size', batch_size)
    writer.add_scalar('param/patch_size', patch_size)
    writer.add_scalar('param/num_epochs', num_epochs)
    if filter_species_ids is not None:
        writer.add_text('param/filter_species_ids', str(filter_species_ids))

    train_dataset, train_dataset_with_non_random_transformation, \
        test_dataset = CUBDataset.load_dataset_splits(
        CUB_DATA_DIR, input_image_size, filter_species_ids)

    num_attributes = train_dataset.num_attributes
    all_attribute_labels = range(1, num_attributes + 1)
    attribute_names = [train_dataset.get_attribute(al).name
                       for al in all_attribute_labels]

    train_dataloader = DataLoader(
        train_dataset, collate_fn=CUBDataset.custom_collate_fn,
        batch_size=batch_size, shuffle=True)

    model = _make_cuda(SemanticAutoencoder(
        input_image_size, patch_size, train_dataset.num_attributes))

    criterion = _make_cuda(CustomLoss2())
    optimizer = optim.Adam(ifilter(lambda p: p.requires_grad, model.parameters()))

    pbar, steps = tqdm(range(1, num_epochs + 1)), 0
    for epoch in pbar:
        epoch_loss = 0.

        for image, label, attribute_labels, padding_idx in train_dataloader:
            steps += 1
            model.zero_grad()

            x = _make_cuda(torch.autograd.Variable(image))
            z, z_patches, reconstructed_x = model(x)

            prototype_labels = _make_cuda(attribute_labels)
            positive_prototypes = model.prototypes(prototype_labels)

            negative_prototypes = list()
            for img_al in attribute_labels:
                negative_al = _make_cuda(torch.LongTensor(list(filter(
                    lambda al: al not in img_al,
                    all_attribute_labels))))
                negative_prototypes.append(model.prototypes(negative_al))

            loss = criterion(reconstructed_x, z_patches,
                             positive_prototypes, padding_idx, x,
                             negative_prototypes=negative_prototypes)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            writer.add_scalar('loss/step_loss', loss, steps)

        writer.add_scalar('loss/epoch_loss', epoch_loss, steps)
        pbar.set_description('epoch_loss=%0.05f' % epoch_loss)

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

        if epoch == num_epochs:
            writer.add_embedding(
                model.prototypes.weight[1:],
                metadata=attribute_names,
                global_step=steps)
    model.save_weights(os.path.join(MODEL_DIR, 'model.pt'))
    writer.close()


if __name__ == '__main__':
    main()
