import os

from absl import app, flags, logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from goggles.constants import *
from goggles.models.vgg import Vgg16
from goggles.opts import DATASET_MAP, DATA_DIR_MAP
from goggles.utils.notify import notify


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'layer_idx', 30,
    'Layer index from the VGG-16 model '
    'to be used for extracting proposals')
flags.DEFINE_enum(
    'dataset', None, ['awa2', 'cub'],
    'Dataset for analysis')
flags.DEFINE_list(
    'class_ids', None,
    'Comma separated class IDs')
flags.DEFINE_integer(
    'num_max_proposals', 5,
    'Max. number of proposals '
    'to be extracted for each image')

flags.mark_flag_as_required('dataset')
flags.mark_flag_as_required('class_ids')

_make_cuda = lambda x: x.cuda() if torch.cuda.is_available() else x


class Context:
    def __init__(self, model, dataset, layer_idx):
        self.model = model
        self.dataset = dataset

        self._layer_idx = layer_idx
        self._model_out_dict = dict()

    def get_model_output(self, image_idx):
        if image_idx not in self._model_out_dict:
            x = self.dataset[image_idx][0]
            x = x.view((1,) + x.size())
            x = _make_cuda(torch.autograd.Variable(x, requires_grad=False))

            self._model_out_dict[image_idx] = \
                self.model.forward(x, layer_idx=self._layer_idx)[0]

        z = self._model_out_dict[image_idx]
        return z


def _get_patches(z, patch_idxs, normalize=False):
    """
    z: CxHxW
    patch_idxs: K
    """

    c = z.size(0)

    patches = z.view(c, -1).t()[patch_idxs]
    if normalize:
        patches = F.normalize(patches, dim=1)

    return patches


def _get_most_activated_channels(z, num_channels=5):
    """
    z: CxHxW
    """

    per_channel_max_activations, _ = z.max(1)[0].max(1)

    most_activated_channels = \
        torch.topk(per_channel_max_activations, num_channels)[1]

    return most_activated_channels


def _get_most_activated_patch_idxs_from_channels(z, channel_idxs):
    """
    z: CxHxW
    channel_idxs: K
    """

    k = channel_idxs.shape[0]

    most_activated_patch_idxs = \
        z[channel_idxs].view(k, -1).max(1)[1]

    return torch.unique(most_activated_patch_idxs)


def _get_score_matrix_for_image(image_idx, num_max_proposals, context):
    score_matrix = list()
    column_ids = list()

    z = context.get_model_output(image_idx)
    num_patches = z.size(1) * z.size(2)
    ch = _get_most_activated_channels(z, num_channels=num_max_proposals)
    pids = _get_most_activated_patch_idxs_from_channels(z, ch)
    proto_patches = _get_patches(z, pids, normalize=True)

    for patch_idx in pids.cpu().numpy():
        column_ids.append([image_idx, patch_idx])

    for image_idx_ in trange(len(context.dataset), leave=True):
        z_ = context.get_model_output(image_idx_)
        img_patches = _get_patches(z_, range(num_patches), normalize=True)
        scores = torch.matmul(img_patches, proto_patches.t()).max(0)[0]
        scores = scores.cpu().numpy()

        score_matrix.append(scores)

    return np.array(score_matrix), column_ids


def main(argv):
    del argv  # unused

    filter_class_ids = list(map(int, FLAGS.class_ids))
    logging.info('calculating scores for classes %s from %s'
                 % (', '.join(map(str, filter_class_ids)),
                    FLAGS.dataset))

    dataset_class = DATASET_MAP[FLAGS.dataset]
    data_dir = DATA_DIR_MAP[FLAGS.dataset]

    logging.info('loading data...')
    input_image_size = 224
    dataset = dataset_class.load_all_data(
        data_dir, input_image_size, filter_class_ids)

    logging.info('loaded %d images' % len(dataset))

    logging.info('loading model...')
    model = _make_cuda(Vgg16())

    context = Context(
        model=model,
        dataset=dataset,
        layer_idx=FLAGS.layer_idx)

    out_filename = '-'.join([
        f'vgg16_layer{FLAGS.layer_idx}',
        FLAGS.dataset,
        '_'.join(map(str, filter_class_ids)),
        'scores.npz'])
    out_dirpath = os.path.join(SCRATCH_DIR, 'scores')
    os.makedirs(out_dirpath, exist_ok=True)
    out_filepath = os.path.join(out_dirpath, out_filename)

    logging.info('saving output to %s' % out_filepath)

    all_scores_matrix = None
    all_column_ids = list()
    for image_idx in trange(len(context.dataset)):
        if all_scores_matrix is None:
            scores, cols = _get_score_matrix_for_image(
                image_idx, FLAGS.num_max_proposals, context)

            all_scores_matrix = scores
            all_column_ids += cols
        else:
            scores, cols = _get_score_matrix_for_image(
                image_idx, FLAGS.num_max_proposals, context)

            all_scores_matrix = np.concatenate(
                (all_scores_matrix, scores), axis=1)
            all_column_ids += cols

        np.savez(out_filepath, scores=all_scores_matrix, cols=all_column_ids)

    notify(f'Finished: {FLAGS.dataset} - {FLAGS.class_ids}', namespace='scores')


if __name__ == '__main__':
    app.run(main)
