import os

import torch

from constants import *
from models.semantic_ae import SemanticAutoencoder
from train import load_datasets
from utils.vis import save_prototype_patch_visualization


input_image_size = 128
filter_species_ids = [14, 90]
patch_size = 1

_, train_dataset_deterministic, _ = load_datasets(input_image_size, CUB_DATA_DIR, filter_species_ids)

model = SemanticAutoencoder(
    input_image_size, patch_size,
    train_dataset_deterministic.num_attributes)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model.pt')).state_dict())
if torch.cuda.is_available():
    model.cuda()

nearest_patches_for_prototypes = \
    model.get_nearest_patches_for_prototypes(
        train_dataset_deterministic)

save_prototype_patch_visualization(
    model, train_dataset_deterministic, save_prototype_patch_visualization, '../out/prototypes/')
