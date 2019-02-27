import numpy as np
import os
import sys
import random

from absl import app, flags, logging
from scipy import sparse
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

SNORKEL_LIB_DIR = '/home/goggles/snorkel'
sys.path.append(SNORKEL_LIB_DIR)
import snorkel
from snorkel.learning import GenerativeModel


GOGGLES_LIB_DIR = '/home/goggles/GOGGLES'
sys.path.append(GOGGLES_LIB_DIR)

from goggles.constants import *
from goggles.data.awa2.dataset import AwA2Dataset
from goggles.data.cub.dataset import CUBDataset
from goggles.utils.notify import notify

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'run', 0,
    'Run ID for this experiment')

flags.DEFINE_enum(
    'dataset', None, ['awa2', 'cub'],
    'Dataset for analysis')

flags.DEFINE_list(
    'class_ids', None,
    'Comma separated class IDs')


def load_scores(filename):
    data = np.load(filename)
    scores_matrix = data['scores']
    col_indices = data['cols']

    return scores_matrix, col_indices


def train_snorkel_gen_model(L_tr, L_te, gte=True):
    L_train = sparse.csr_matrix(L_tr)
    L_test = sparse.csr_matrix(L_te)
    gen_model = GenerativeModel()
    gen_model.train(L_train, epochs=100, decay=0.95,
                    step_size=0.01 / L_train.shape[0],
                    reg_param=1e-6)

    test_marginals = gen_model.marginals(L_test)
    marginals_threshold = (max(test_marginals) - min(test_marginals)) / 2
    pred_labels = (2 * (test_marginals >= marginals_threshold) - 1 if gte
                   else 2 * (test_marginals < marginals_threshold) - 1)

    return gen_model, pred_labels, test_marginals


def get_labeling_matrix_for_GOOGGLES(scores_matrix):
    num_cols = scores_matrix.shape[1]
    col_labels = KMeans(n_clusters=2).fit_predict(scores_matrix.T)
    col_labels[np.where(col_labels == 0)] = -1

    L = list()
    for j in range(num_cols):
        col = scores_matrix[:, j]
        col_label = col_labels[j]
        threshold = (np.max(col) + np.min(col)) / 2.0

        col_lf = [col_label if v > threshold else 0 for v in col]

        L.append(col_lf)
    L = np.array(L).T
    return L


def best_acc(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    return max(acc, 1. - acc)



def main(argv):
    del argv  # unused

    preds_out_dirpath = os.path.join(SCRATCH_DIR, 'preds_snorkel')
    os.makedirs(preds_out_dirpath, exist_ok=True)
    class_ids = list(map(int, FLAGS.class_ids))

    preds_out_filename = '-'.join([
        FLAGS.dataset,
        '_'.join(map(str, class_ids)),
        'run_%02d' % FLAGS.run,
        'preds_snorkel.npz'])

    preds_out_filepath = os.path.join(
        preds_out_dirpath, preds_out_filename)

    assert not os.path.exists(preds_out_filepath), \
        'Predictions for this run already exists at %s' % preds_out_filepath

    input_image_size = 224

    if FLAGS.dataset == 'cub':
        dataset = CUBDataset.load_all_data(
            CUB_DATA_DIR, input_image_size, class_ids)
    elif FLAGS.dataset == 'awa2':
        dataset = AwA2Dataset.load_all_data(
            AWA2_DATA_DIR, input_image_size, class_ids)

    y_true = [v[1] for v in dataset]

    seed = sum(v * (10 ** (3 * i))
               for i, v in enumerate(class_ids + [FLAGS.run]))
    random.seed(seed)
    np.random.seed(seed)

    scores, col_ids = load_scores(
        os.path.join(
            SCRATCH_DIR, 'scores',
            f'vgg16_layer30-{FLAGS.dataset}-%d_%d-scores.npz'
            % tuple(class_ids)))

    new_scores_np = get_labeling_matrix_for_GOOGGLES(scores)

    L_tr, L_te = new_scores_np, new_scores_np
    _, y_snorkel, _ = train_snorkel_gen_model(L_tr, L_te)

    np.savez(preds_out_filepath,
             y_true=y_true, y_snorkel=y_snorkel)

    logging.info(f'saved predictions at {preds_out_filepath}')

    snorkel_acc = best_acc(y_true, y_snorkel)

    # notify(f'`{FLAGS.dataset}` - `%s` - `run {FLAGS.run}`: '
    #        f'{snorkel_acc}'
    #        % ', '.join(map(str, class_ids)),
    #        namespace='goggles-snorkel')


if __name__ == '__main__':
    app.run(main)
