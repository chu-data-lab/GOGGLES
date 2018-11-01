import os

from joblib import Memory
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from goggles.constants import CACHE_DIR
from metadata import load_cub_metadata


class CUBDataset(Dataset):
    def __init__(self, root, species1_id, species2_id, transform=None, is_training=False, cachedir=CACHE_DIR):
        super(CUBDataset, self).__init__()

        mem = Memory(cachedir)
        metadata_loader = mem.cache(load_cub_metadata)

        self._data_dir = root
        all_species, all_attributes, all_images_data = metadata_loader(root)

        self._species1 = filter(lambda s: s.id == species1_id, all_species)[0]
        self._species2 = filter(lambda s: s.id == species2_id, all_species)[0]

        self._image_data = list(sorted(
            filter(
                lambda datum: (
                    datum.species in [self._species1, self._species2]
                    and datum.is_for_training == is_training),
                all_images_data),
            key=lambda d: d.id))

        self._attributes = list(sorted(
            filter(lambda attr: attr in self._species1.attributes.union(self._species2.attributes), all_attributes),
            key=lambda attr: attr.id))
        self.num_attributes = len(self._attributes)

        if transform is not None:
            self._transform = transform
        else:
            self._transform = transforms.Compose([transforms.ToTensor()])

        self.is_training = is_training

    def __len__(self):
        return len(self._image_data)

    def __getitem__(self, idx):
        datum = self._image_data[idx]

        image_file = os.path.join(self._data_dir, 'CUB_200_2011', 'images', datum.path)
        image = Image.open(image_file)
        image = self._transform(image)

        label = 0 if datum.species.name == self._species1.name else 1

        attributes = list()
        for attr in datum.attribute_annotations:
            if attr in self._attributes:
                attributes.append(self._attributes.index(attr) + 1)  # attributes are 1-indexed
        num_nonzero_attributes = len(attributes)
        attributes = sorted(attributes) + ([0] * (self.num_attributes - len(attributes)))  # 0's added for padding

        return image, label, attributes, num_nonzero_attributes

    @staticmethod
    def custom_collate_fn(batch):
        batch = zip(*batch)  # transpose

        image, label, attributes, \
            num_nonzero_attributes = batch

        image = torch.stack(image)
        label = torch.LongTensor(label)
        attributes = torch.stack([torch.LongTensor(a) for a in attributes])
        padding_idx = torch.LongTensor(num_nonzero_attributes)

        return image, label, attributes, padding_idx

    def get_attribute_name_for_attribute_idx(self, attribute_idx):
        return self._attributes[attribute_idx - 1].name  # attributes are 1-indexed


if __name__ == '__main__':
    data_dir = '/Users/nilakshdas/Dev/GOGGLES/data/CUB_200_2011'

    train_dataset = CUBDataset(data_dir, 14, 90, is_training=True)
    test_dataset = CUBDataset(data_dir, 14, 90, is_training=True)

    assert train_dataset.num_attributes == test_dataset.num_attributes
    for i in range(train_dataset.num_attributes):
        assert train_dataset._attributes[i] == test_dataset._attributes[i]
        print train_dataset._attributes[i], test_dataset._attributes[i]

    # for d, l, a, _ in dataset:
    #     print d.size(), l, a
