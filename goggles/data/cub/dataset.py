import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from metadata import load_cub_metadata


class CUBDataset(Dataset):
    def __init__(self, root, species1_id, species2_id, transform=None, is_training=False):
        super(CUBDataset, self).__init__()

        self._data_dir = root
        all_species, all_attributes, all_images_data = load_cub_metadata(root)

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

        label = 0 if datum.species is self._species1 else 1

        attributes = list()
        for attr in datum.attribute_annotations:
            if attr in self._attributes:
                attributes.append(self._attributes.index(attr))
        attributes = sorted(attributes)

        return image, label, attributes


if __name__ == '__main__':
    from goggles.constants import *

    dataset = CUBDataset(CUB_DATA_DIR, 1, 2)
    for d, l, a in dataset:
        print d.size(), l, a
