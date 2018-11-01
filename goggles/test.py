import os

import torch

from constants import *
from train import load_datasets
from utils.vis import save_prototype_patch_visualization


input_image_size = 64
species1_id = 14
species2_id = 90
patch_size = 1

_, _, test_dataset = load_datasets(input_image_size, CUB_DATA_DIR, species1_id, species2_id)

model = torch.load(os.path.join(MODEL_DIR, 'model.pt'))

save_prototype_patch_visualization(model, test_dataset, '../out/prototypes/')
