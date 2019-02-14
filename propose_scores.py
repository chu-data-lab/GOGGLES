import os
from types import SimpleNamespace

from absl import app, flags, logging
import numpy as np
from scipy import spatial
import torch
from tqdm import tqdm, trange

from goggles.constants import *
from goggles.models.patch import Patch
from goggles.models.vgg import Vgg16
from goggles.opts import DATASET_MAP, DATA_DIR_MAP


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
    'filter_class_label', None,
    'The class label for which '
    'the scores are to be computed')
flags.DEFINE_integer(
    'num_proposals', 5,
    'Number of proposals to be '
    'extracted for each image')

flags.mark_flag_as_required('dataset')
flags.mark_flag_as_required('class_ids')

_make_cuda = lambda x: x.cuda() if torch.cuda.is_available() else x


def _get_embedding_for_patch_id(patch_id, context):
    image_idx, patch_idx = patch_id

    x = context.dataset[image_idx][0]
    x = x.view((1,) + x.size())
    x = _make_cuda(torch.autograd.Variable(x, requires_grad=False))

    z = context.model.forward(x, layer_idx=context.layer_idx)
    e = context.patches[patch_idx].forward(z).cpu().numpy()[0]

    return e


def _get_proposed_patch_ids_for_image(image_idx, num_max_proposals, context):
    x = context.dataset[image_idx][0]
    x = x.view((1,) + x.size())
    x = _make_cuda(torch.autograd.Variable(x, requires_grad=False))

    z = context.model.forward(x, layer_idx=context.layer_idx)
    z_np = z[0].cpu().numpy()
    
    best_channel_indices = \
        np.max(z_np, axis=(1, 2)).argsort()[::-1][:num_max_proposals]
    
    proposed_patch_ids = set()
    for i in range(num_max_proposals):
        ch = z_np[best_channel_indices[i]]

        most_activated_patch_idx = np.argmax(ch)
        proposed_patch_ids.add((image_idx, most_activated_patch_idx,))

    return list(sorted(proposed_patch_ids))


def _get_score_matrix_for_image(image_idx, num_max_proposals, context):
    dist_fn = spatial.distance.cosine
    score_matrix = list()

    proposed_patch_ids = _get_proposed_patch_ids_for_image(
        image_idx, num_max_proposals, context)
    for patch_id in tqdm(proposed_patch_ids, leave=True):
        init_emb = _get_embedding_for_patch_id(patch_id, context)
        proposal_scores = list()

        for image_idx_ in trange(len(context.dataset), leave=True):
            x = context.dataset[image_idx_][0]
            x = x.view((1,) + x.size())
            x = _make_cuda(torch.autograd.Variable(x, requires_grad=False))

            z = context.model.forward(x, layer_idx=context.layer_idx)

            nearest_patch = min(context.patches, key=lambda patch:
                dist_fn(init_emb, patch.forward(z)[0].cpu().numpy()))

            sim = dist_fn(init_emb, nearest_patch.forward(z)[0].cpu().numpy())
            proposal_scores.append(sim)

        score_matrix.append(proposal_scores)

    return np.array(score_matrix).T


def main(argv):
    del argv  # unused

    filter_class_ids = list(map(int, FLAGS.class_ids))
    logging.info('calculating scores for classes %s'
                 % ', '.join(map(str, filter_class_ids)))

    dataset_class = DATASET_MAP[FLAGS.dataset]
    data_dir = DATA_DIR_MAP[FLAGS.dataset]

    logging.info('loading data...')
    input_image_size = 224
    _, train_dataset, test_dataset = dataset_class.load_dataset_splits(
        data_dir, input_image_size, filter_class_ids)

    train_dataset.merge_image_data(test_dataset)
    logging.info('loaded %d images' % len(train_dataset))

    logging.info('loading model...')
    model = _make_cuda(Vgg16())
    patch_size = (1, 1,)
    encoded_output_dim = model.get_layer_output_dim(FLAGS.layer_idx)
    patches = Patch.from_spec(encoded_output_dim, patch_size)

    context = SimpleNamespace(
        model=model,
        patches=patches,
        dataset=train_dataset,
        layer_idx=FLAGS.layer_idx)

    out_filename = 'label-%s.npy' % (FLAGS.filter_class_label
                                     if FLAGS.filter_class_label is not None
                                     else 'all')
    out_dirpath = os.path.join(SCRATCH_DIR, 'scores', FLAGS.dataset,
                               f'vgg16-layer{FLAGS.layer_idx}',
                               '_'.join(map(str, filter_class_ids)))
    out_filepath = os.path.join(out_dirpath, out_filename)

    os.makedirs(out_dirpath, exist_ok=True)
    logging.info('saving output to %s' % out_filepath)

    all_scores_matrix = None
    for image_idx in trange(len(context.dataset)):
        image_label = context.dataset[image_idx][1]
        if (FLAGS.filter_class_label is None
                or image_label == FLAGS.filter_class_label):

            if all_scores_matrix is None:
                all_scores_matrix = _get_score_matrix_for_image(
                    image_idx, FLAGS.num_proposals, context)
            else:
                all_scores_matrix = np.concatenate(
                    (all_scores_matrix, _get_score_matrix_for_image(
                        image_idx, FLAGS.num_proposals, context)), axis=1)

            np.save(out_filepath, all_scores_matrix)


if __name__ == '__main__':
    app.run(main)
