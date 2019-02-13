from absl import app, flags, logging


FLAGS = flags.FLAGS

flags.DEFINE_list('class_ids', None,
                  'Comma separated class IDs')
flags.DEFINE_integer('image_id', None,
                     'Image ID')

flags.DEFINE_string('dataset', None,
                  'Dataset name')

flags.mark_flag_as_required('class_ids')
flags.mark_flag_as_required('image_id')

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import torch
from torchvision import transforms

from goggles.constants import *
from goggles.data.cub.dataset import CUBDataset
from goggles.data.awa2.dataset import AwA2Dataset
from goggles.models.patch import Patch
from goggles.models.vgg import Vgg16
from goggles.opts import DATASET_MAP, DATA_DIR_MAP
from goggles.utils.vis import get_image_from_tensor, get_image_with_patch_outline

LAYER_IDX = 30
DIST_FN = spatial.distance.cosine
NUM_SKIP_PATCHES = 40


def get_embedding_for_patch_id(patch_id, dataset, model, patches, layer_idx):
    image_idx, patch_idx = patch_id

    x = dataset[image_idx][0]
    x = x.view((1,) + x.size())
    x = torch.autograd.Variable(x, requires_grad=False)

    z = model.forward(x, layer_idx=layer_idx)

    e = patches[patch_idx].forward(z).numpy()[0]

    return e


def get_proposed_patch_ids_for_image(image_idx, num_proposals, dataset, model, patches):
    x = dataset[image_idx][0]
    x = x.view((1,) + x.size())
    x = torch.autograd.Variable(x, requires_grad=False)

    z = model.forward(x, layer_idx=LAYER_IDX)
    z_np = z[0].numpy()

    best_channel_indices = np.max(z_np, axis=(1, 2)).argsort()[::-1][:num_proposals]

    proposed_patch_ids = set()
    for i in range(num_proposals):
        ch = z_np[best_channel_indices[i]]

        most_activated_patch_idx = np.argmax(ch)
        proposed_patch_ids.add((image_idx, most_activated_patch_idx,))

    return list(sorted(proposed_patch_ids))


def get_initial_property_matrix(image_list, test_dataset, ctx, model, patches):
    dists_mat = list()
    for patch_image_id in image_list:

        num_best_channels = 5

        for i, patch_id in enumerate(get_proposed_patch_ids_for_image(patch_image_id, num_best_channels, *ctx)):

            image_idx = patch_id[0]
            patch_idx = patch_id[1]
            INIT_EMB = get_embedding_for_patch_id((image_idx, patch_idx,), *ctx)
            long_list = list()

            for image_id_ in range(len(test_dataset)):
                x_ = test_dataset[image_id_][0]
                x_ = x_.view((1,) + x_.size())
                x_ = torch.autograd.Variable(x_, requires_grad=False)

                z_ = model.forward(x_, layer_idx=LAYER_IDX)

                nearest_patch_idx, nearest_patch = min(
                    enumerate(patches),
                    key=lambda (i, patch): DIST_FN(INIT_EMB, patch.forward(z_)[0].numpy()))

                sim = DIST_FN(INIT_EMB, nearest_patch.forward(z_)[0].numpy())

                long_list.append(sim)


            dists_mat.append(long_list)

    return np.array(dists_mat).T

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def main(argv):
    del argv # unused

    filter_class_ids = list(map(int, FLAGS.class_ids))
    image_id = FLAGS.image_id
    dataset = FLAGS.dataset

    Dataset = DATASET_MAP[dataset]
    data_dir = DATA_DIR_MAP[dataset]

    # logging.info((filter_class_ids, image_id, dataset))

    model = Vgg16()
    input_image_size = 224
    _, train_dataset, test_dataset = Dataset.load_dataset_splits(
        data_dir, input_image_size, filter_class_ids)
    train_dataset.make_balanced_dataset()
    test_dataset.make_balanced_dataset()


    patch_size = (1, 1,)
    encoded_output_dim = model.get_layer_output_dim(LAYER_IDX)
    patches = Patch.from_spec(encoded_output_dim, patch_size)

    ctx = (test_dataset, model, patches, LAYER_IDX)

    np_property_matrix = get_initial_property_matrix(image_id, test_dataset, ctx, model, patches)

    plt.imshow(np_property_matrix)


if __name__ == '__main__':
    app.run(main)

