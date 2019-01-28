from goggles.data.dataset import GogglesDataset
from goggles.data.awa2.metadata import load_awa2_metadata


class AwA2Dataset(GogglesDataset):
    def _load_metadata(self, root_dir):
        return load_awa2_metadata(root_dir)


if __name__ == '__main__':
    # from torch.utils.data import DataLoader

    data_dir = '/media/seagate/rtorrent/AwA2/'

    train_dataset, _, _ = AwA2Dataset.load_dataset_splits(
        data_dir, 224, filter_species_ids=[3, 27])

    # train_dataloader = DataLoader(
    #     train_dataset, collate_fn=AwA2Dataset.custom_collate_fn,
    #     batch_size=4, shuffle=True)
    #
    # for image, label, batch_attribute_labels, padding_idx in train_dataloader:
    #     for image_attribute_labels in batch_attribute_labels:
    #         print(image_attribute_labels[:10])
    #         print(torch.LongTensor(list(filter(
    #             lambda al: al not in image_attribute_labels,
    #             all_attribute_labels))))
    #         print('---')




