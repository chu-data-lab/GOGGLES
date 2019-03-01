from collections import Counter
from itertools import permutations
from joblib import Parallel, delayed
import multiprocessing
import os
import pickle
import random

from absl import app, flags, logging
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from goggles.constants import *
from goggles.data.awa2.dataset import AwA2Dataset
from goggles.data.cub.dataset import CUBDataset
from goggles.utils.notify import notify


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'run', None,
    'Run ID for this experiment')
flags.DEFINE_enum(
    'dataset', None, ['awa2', 'cub'],
    'Dataset for analysis')
flags.DEFINE_list(
    'class_ids', None,
    'Comma separated class IDs')

flags.mark_flag_as_required('run')
flags.mark_flag_as_required('dataset')
flags.mark_flag_as_required('class_ids')


def load_scores(filename, pick_one_prototype=False):
    data = np.load(filename)
    scores_matrix = data['scores']
    col_indices = data['cols']

    if pick_one_prototype:
        j_list = sorted({i: j for j, (i, p) in
                         enumerate(col_indices)}.values())

        scores_matrix = scores_matrix[:, j_list]
        col_indices = col_indices[j_list]
            
    return scores_matrix, col_indices


def best_acc(y_true, y_pred):
    gt_labels = list(sorted(set(y_true)))
    pred_labels = list(sorted(set(y_pred)))

    n_gt_labels = len(gt_labels)
    n_pred_labels = len(pred_labels)

    if n_pred_labels < n_gt_labels:
        return 0.

    accs = list()
    perms = list(permutations(pred_labels, n_gt_labels))
    for permutation in tqdm(perms):
        labeling = dict(zip(permutation, gt_labels))

        new_y_pred = list()
        for i, pred in enumerate(y_pred):
            if pred in labeling.keys():
                new_y_pred.append(labeling[pred])
            else:
                new_y_pred.append(max(gt_labels) + 1)

        accs.append(accuracy_score(y_true, new_y_pred))

    return max(accs)


class GogglesProbabilisticModel:
    class Gaussian:
        def __init__(self, mu, std):
            self.mu = mu
            self.std = std

        def plot(self, axis):
            x = np.linspace(0, 1, 1000)
            pdf = norm.pdf(x, self.mu, self.std)
            axis.plot(x, pdf, linewidth=4)

        def log_cdf(self, s):
            return norm.logcdf(s, loc=self.mu, scale=self.std)

        def log_sf(self, s):
            return norm.logsf(s, loc=self.mu, scale=self.std)

    def __init__(self, scores, cols, y, p1=None):
        self._scores = scores
        self._cols = cols
        self._y = y

        self._num_rows = scores.shape[0]
        self._num_cols = scores.shape[1]
        self._labels = list(sorted(np.unique(y)))

        assert p1 is None, 'p1 is not used anymore'
        self._p1 = p1

        counts = Counter(list(y))
        self._p = {label: counts[label] / float(len(y))
                   for label in self._labels}

        assert len(y) == self._num_rows

        self._params = list()
        for j in range(self._num_cols):
            self._params.append(self.fit_conditional_parameters(j))

    @property
    def labels(self):
        return self._labels

    def y(self, i):
        return self._y[i]

    def z(self, j):
        i, _ = self._cols[j]
        return self.y(i)

    def s(self, i, j):
        return self._scores[i, j]

    def log_alpha_k(self, j, k, s):
        gaussian = self._params[j][k]

        if self.z(j) == k:
            return gaussian.log_cdf(s)
        else:
            return gaussian.log_sf(s)

    def log_a_k(self, i, k):
        log_ai = 0.

        for j in range(self._num_cols):
            sij = self.s(i, j)
            log_ai += self.log_alpha_k(j, k, sij)

        return log_ai

    def gamma(self, i):
        log_ai = {k: self.log_a_k(i, k) for k in self._labels}

        gam = dict()
        for k in self._labels:
            pk = self._p[k]

            den = 0.
            for k_ in self._labels:
                if k_ == k:
                    den += pk
                else:
                    pk_ = self._p[k_]
                    bi_over_ai = np.exp(
                        max(min(log_ai[k_] - log_ai[k], 700), -700))
                    den += pk_ * bi_over_ai

            gam[k] = pk / den

        return gam

    def get_class_wise_scores(self, j):
        class_wise_scores = dict()
        for label in self._labels:
            class_wise_scores[label] = \
                self._scores[np.where(self._y == label), j].flatten()

        return class_wise_scores

    def fit_conditional_parameters(self, j):
        class_wise_scores = self.get_class_wise_scores(j)

        class_wise_parameters = dict()
        for label in self._labels:
            gmm = GaussianMixture(n_components=1)
            gmm.fit(class_wise_scores[label].reshape(-1, 1))

            class_wise_parameters[label] = \
                self.Gaussian(mu=gmm.means_.flatten()[0],
                              std=np.sqrt(gmm.covariances_.flatten()[0]))

        return class_wise_parameters

    def update_model(self, y, update_prior=False):
        self.__init__(self._scores, self._cols, y,
                      p1=(None if update_prior
                          else self._p1))

    def save_model(self, filepath):
        pickle.dump(self, open(filepath, 'wb'))

    @staticmethod
    def load_model(filepath):
        return pickle.load(open(filepath, 'rb'))

    @classmethod
    def run_em(cls, scores, cols, y_init,
               p1=None, update_prior=False,
               max_iter=100):

        n = y_init.shape[0]
        y = np.array(y_init)

        num_cores = multiprocessing.cpu_count()

        model = cls(scores, cols, y, p1=p1)
        with tqdm(range(max_iter), leave=True) as pbar:
            for _ in pbar:
                # E-step
                y_new = list()

                if n >= 500:
                    gamma = Parallel(n_jobs=num_cores)(
                        delayed(model.gamma)(i) for i in range(n))
                else:
                    gamma = [model.gamma(i)
                             for i in tqdm(range(n), leave=True)]

                for i in range(n):
                    gamma_i = gamma[i]
                    probs = list(map(lambda it: it[1],
                                     sorted(gamma_i.items(),
                                            key=lambda it: it[0])))

                    y_i = np.random.choice(len(model.labels), 1, p=probs)[0]
                    y_new.append(y_i)

                # M-step
                y_new = np.array(y_new)
                model.update_model(y, update_prior=update_prior)

                if np.linalg.norm(y_new - y) == 0:
                    break
                y = np.array(y_new)

        return model, y_new


def main(argv):
    del argv  # unused

    preds_out_dirpath = os.path.join(SCRATCH_DIR, 'preds-multi')
    models_out_dirpath = os.path.join(SCRATCH_DIR, 'models-multi')
    os.makedirs(preds_out_dirpath, exist_ok=True)
    os.makedirs(models_out_dirpath, exist_ok=True)

    class_ids = list(map(int, FLAGS.class_ids))

    preds_out_filename = '-'.join([
        FLAGS.dataset,
        '_'.join(map(str, class_ids)),
        'run_%02d' % FLAGS.run,
        'preds.npz'])
    kmeans_init_model_out_filename = '-'.join([
        FLAGS.dataset,
        '_'.join(map(str, class_ids)),
        'run_%02d' % FLAGS.run,
        'kmeans_init',
        'model.pkl'])
    rand_init_model_out_filename = '-'.join([
        FLAGS.dataset,
        '_'.join(map(str, class_ids)),
        'run_%02d' % FLAGS.run,
        'rand_init',
        'model.pkl'])

    preds_out_filepath = os.path.join(
        preds_out_dirpath, preds_out_filename)
    kmeans_init_model_out_filepath = os.path.join(
        models_out_dirpath, kmeans_init_model_out_filename)
    rand_init_model_out_filepath = os.path.join(
        models_out_dirpath, rand_init_model_out_filename)

    assert not os.path.exists(preds_out_filepath), \
        'Predictions for this run already exists at %s' % preds_out_filepath
    assert not os.path.exists(kmeans_init_model_out_filepath), \
        'Model (k-means init) for this run already exists at %s' % \
        kmeans_init_model_out_filepath
    assert not os.path.exists(rand_init_model_out_filepath), \
        'Model (random init) for this run already exists at %s' % \
        rand_init_model_out_filepath

    logging.info(f'calculating run {FLAGS.run} accuracies '
                 f'for classes %s from %s' % (', '.join(map(str, class_ids)),
                                              FLAGS.dataset))

    input_image_size = 224

    if FLAGS.dataset == 'cub':
        dataset = CUBDataset.load_all_data(
            CUB_DATA_DIR, input_image_size, class_ids)
    elif FLAGS.dataset == 'awa2':
        dataset = AwA2Dataset.load_all_data(
            AWA2_DATA_DIR, input_image_size, class_ids)

    y_true = [v[1] for v in dataset]
    
    scores, col_ids = load_scores(
        os.path.join(
            SCRATCH_DIR, 'scores',
            f'vgg16_layer30-{FLAGS.dataset}-%s-scores.npz'
            % '_'.join(map(str, class_ids))),
        pick_one_prototype=True)

    seed = sum(v * (3 ** (3 * i))
               for i, v in enumerate(class_ids + [FLAGS.run]))
    random.seed(seed)
    np.random.seed(seed)

    num_classes = len(class_ids)
    logging.info(f'performing {num_classes}-class labeling...')

    y_kmeans = KMeans(n_clusters=num_classes).fit_predict(scores)
    kmeans_acc = best_acc(y_true, y_kmeans)
    
    try:
        kmeans_init_model, y_kmeans_em = \
            GogglesProbabilisticModel.run_em(scores, col_ids, y_kmeans,
                                             p1=None, update_prior=True)

        kmeans_init_model.save_model(kmeans_init_model_out_filepath)
        logging.info(f'saved k-means init model at '
                     f'{kmeans_init_model_out_filepath}')

        kmeans_em_acc = best_acc(y_true, y_kmeans_em)
    except Exception as e:
        print(e)
        kmeans_em_acc = 0.

    logging.info('image counts: %s' % str(Counter(y_true)))

    logging.info('only kmeans accuracy for classes %s: %0.9f' 
                 % (', '.join(map(str, class_ids)), 
                    kmeans_acc))
    logging.info('kmeans + em accuracy for classes %s: %0.9f' 
                 % (', '.join(map(str, class_ids)), 
                    kmeans_em_acc))

    np.savez(preds_out_filepath,
             y_true=y_true, y_kmeans=y_kmeans,
             y_kmeans_em=y_kmeans_em)
    logging.info(f'saved predictions at {preds_out_filepath}')

    notify(f'`{FLAGS.dataset}` - `%s` - `run {FLAGS.run}`: '
           f'{kmeans_acc}, {kmeans_em_acc}'
           % ', '.join(map(str, class_ids)),
           namespace='inference-multi')


if __name__ == '__main__':
    app.run(main)
