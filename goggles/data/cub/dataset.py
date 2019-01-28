import torch

from goggles.data.dataset import GogglesDataset
from goggles.data.cub.metadata import load_cub_metadata


class CUBDataset(GogglesDataset):
    def _load_metadata(self, root_dir):
        return load_cub_metadata(root_dir)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_dir = '/Users/sanyachaba/Desktop/GATECH/Fall 2018/Project/GOGGLES/_scratch/CUB_200_2011'

    train_dataset, _, _ = CUBDataset.load_dataset_splits(
        data_dir, 128, filter_species_ids=[14, 90])

    num_attributes = train_dataset.num_attributes
    all_attribute_labels = range(1, num_attributes + 1)

    train_dataloader = DataLoader(
        train_dataset, collate_fn=CUBDataset.custom_collate_fn,
        batch_size=4, shuffle=True)

    for image, label, batch_attribute_labels, padding_idx in train_dataloader:
        for image_attribute_labels in batch_attribute_labels:
            print(image_attribute_labels[:10])
            print(torch.LongTensor(list(filter(
                lambda al: al not in image_attribute_labels,
                all_attribute_labels))))
            print('---')
