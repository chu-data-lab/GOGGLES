import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GogglesDataset(Dataset):
    def __init__(self,path,transform):
        valid_images = [".jpg", ".gif", ".png"]
        self._data_path = path
        self.images_filename_list = []
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            self.images_filename_list.append(f)

        if transform is not None:
            self._transform = transform
        else:
            self._transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        """
        read a image only when it is used
        :param idx: integer
        :return:
        """
        filename = self.images_filename_list[idx]
        try:
            image_file = os.path.join(self._data_path, filename)
            image = Image.open(image_file).convert('RGB')
            image = self._transform(image)
        except:
            image = None
        return image

    def __len__(self):
        return len(self.images_filename_list)

    @classmethod
    def load_all_data(cls, root_dir, input_image_size):
        try:
            transform_resize = transforms.Resize(
                (input_image_size, input_image_size))
        except AttributeError:
            transform_resize = transforms.Scale(
                (input_image_size, input_image_size))

        transform_to_tensor = transforms.ToTensor()
        transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        transformation = transforms.Compose([
            transform_resize, transform_to_tensor, transform_normalize])
        dataset = cls(
            root_dir,
            transform=transformation)
        return dataset