from collections import Counter
from joblib import Parallel, delayed
import multiprocessing
import os
import pickle
import random

from absl import app, flags, logging
from numba import jit, prange
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from goggles.constants import *
from goggles.data.awa2.dataset import AwA2Dataset
from goggles.data.cub.dataset import CUBDataset


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


def load_scores(filename):
    data = np.load(filename)
    scores_matrix = data['scores']
    col_indices = data['cols']
            
    return scores_matrix, col_indices


def best_acc(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    return max(acc, 1. - acc)


class GogglesProbabilisticModel:
    class Gaussian:
        def __init__(self, mu, std):
            self.mu = mu
            self.std = std

        def plot(self, axis):
            x = np.linspace(0, 1, 1000)
            pdf = norm.pdf(x, self.mu, self.std)
            axis.plot(x, pdf, linewidth=4)

        @jit('float64(float64)', nopython=True, nogil=True)
        def log_cdf(self, s):
            return norm.logcdf(s, loc=self.mu, scale=self.std)

        @jit('float64(float64)', nopython=True, nogil=True)
        def log_sf(self, s):
            return norm.logsf(s, loc=self.mu, scale=self.std)
    
    def __init__(self, scores, cols, y, p1=None):
        self._scores = scores
        self._cols = cols
        self._y = y

        if p1 is None:
            p1 = Counter(list(y))[1] / float(len(y))
        self._p1 = p1
        
        self._num_rows = scores.shape[0]
        self._num_cols = scores.shape[1]
        self._labels = list(sorted(np.unique(y)))

        assert len(y) == self._num_rows
        
        self._params = list()
        for j in range(self._num_cols):
            self._params.append(self.fit_conditional_parameters(j))

    @jit('uint32(uint32)', nopython=True, nogil=True)
    def y(self, i):
        return self._y[i]

    @jit('uint32(uint32)', nopython=True, nogil=True)
    def z(self, j):
        i, _ = self._cols[j]
        return self.y(i)

    @jit('float64(uint32, uint32)', nopython=True, nogil=True)
    def s(self, i, j):
        return self._scores[i, j]

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

    @jit('float64(uint32, float64)', nopython=True, nogil=True)
    def log_alpha(self, j, s):
        gaussian = self._params[j][1]
        
        if self.z(j) == 1:
            return gaussian.log_cdf(s)
        elif self.z(j) == 0:
            return gaussian.log_sf(s)
        else:
            raise ValueError('Only binary labels supported')

    @jit('float64(uint32, float64)', nopython=True, nogil=True)
    def log_beta(self, j, s):
        gaussian = self._params[j][0]
        
        if self.z(j) == 0:
            return gaussian.log_cdf(s)
        elif self.z(j) == 1:
            return gaussian.log_sf(s)
        else:
            raise ValueError('Only binary labels supported')

    @jit('float64(uint32)', nopython=True, nogil=True, parallel=True)
    def log_a(self, i):
        log_ai = 0.
        
        for j in prange(self._num_cols):
            sij = self.s(i, j)
            log_ai += self.log_alpha(j, sij)
                
        return log_ai

    @jit('float64(uint32)', nopython=True, nogil=True, parallel=True)
    def log_b(self, i):
        log_bi = 0.
        
        for j in prange(self._num_cols):
            sij = self.s(i, j)
            log_bi += self.log_beta(j, sij)
                
        return log_bi

    @jit('float64(uint32)', nopython=True, nogil=True, parallel=True)
    def tau(self, i):
        log_ai = self.log_a(i)
        log_bi = self.log_b(i)

        p1 = self._p1
        
        bi_over_ai = np.exp(max(min(log_bi - log_ai, 700), -700))
        
        t = p1 / (p1 + ((1 - p1) * bi_over_ai))
        return min(t, 1.)

    def get_probabilistic_labels(self, scores):
        probs = list()
        for i in range(scores.shape[0]):
            log_ai = sum(self.log_alpha(j, scores[i, j])
                         for j in range(scores.shape[1]))
            log_bi = sum(self.log_beta(j, scores[i, j])
                         for j in range(scores.shape[1]))

            p1 = self._p1

            bi_over_ai = np.exp(max(min(log_bi - log_ai, 700), -700))

            t = p1 / (p1 + ((1 - p1) * bi_over_ai))
            probs.append(min(t, 1.))

        return np.array(probs)
    
    def update_model(self, y, update_prior=False):
        self.__init__(self._scores, self._cols, y,
                      p1=(None if update_prior
                          else self._p1))

    def save_model(self, filepath):
        out = open(filepath, 'wb')
        pickle.dump(self, out)

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
                tau = Parallel(n_jobs=num_cores)(
                    delayed(model.tau)(i) for i in range(n))
                for i in range(n):
                    tau_i = tau[i]
                    y_i = np.random.choice(2, 1, p=[1 - tau_i, tau_i])[0]
                    y_new.append(y_i)

                # M-step
                y_new = np.array(y_new)
                model.update_model(y_new, update_prior=update_prior)

                if np.linalg.norm(y_new - y) == 0:
                    break
                y = np.array(y_new)
            
        return model, y_new


def main(argv):
    del argv  # unused

    preds_out_dirpath = os.path.join(SCRATCH_DIR, 'preds')
    models_out_dirpath = os.path.join(SCRATCH_DIR, 'models')
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

    logging.info('calculating accuracies for classes %s from %s'
                 % (', '.join(map(str, class_ids)), FLAGS.dataset))

    input_image_size = 224

    logging.info('loading data...')
    if FLAGS.dataset == 'cub':
        dataset = CUBDataset.load_all_data(
            CUB_DATA_DIR, input_image_size, class_ids)
    elif FLAGS.dataset == 'awa2':
        dataset = AwA2Dataset.load_all_data(
            AWA2_DATA_DIR, input_image_size, class_ids)

    y_true = [v[1] for v in dataset]

    logging.info('loading scores...')
    scores, col_ids = load_scores(
        os.path.join(
            SCRATCH_DIR, 'scores',
            f'vgg16_layer30-{FLAGS.dataset}-%d_%d-scores.npz' 
            % tuple(class_ids)))

    seed = sum(v * (10 ** (3 * i))
            for i, v in enumerate(class_ids + [FLAGS.run]))
    random.seed(seed)
    np.random.seed(seed)

    logging.info('running k-means...')
    y_kmeans = KMeans(n_clusters=2).fit_predict(scores)
    kmeans_acc = best_acc(y_true, y_kmeans)

    logging.info('running EM with k-means initialization...')
    try:
        kmeans_init_model, y_kmeans_em = \
            GogglesProbabilisticModel.run_em(scores, col_ids, y_kmeans)

        kmeans_init_model.save_model(kmeans_init_model_out_filepath)
        logging.info(f'Saved k-means init model at '
                     f'{kmeans_init_model_out_filepath}')

        kmeans_em_acc = best_acc(y_true, y_kmeans_em)
    except:
        kmeans_em_acc = 0.

    logging.info('running EM with random initialization...')
    try:
        y_init = np.random.randint(2, size=scores.shape[0])
        p1 = Counter(list(y_kmeans))[1] / float(len(y_kmeans))

        rand_init_model, y_rand_em = \
            GogglesProbabilisticModel.run_em(scores, col_ids, y_init, p1=p1)
        rand_init_model.save_model(rand_init_model_out_filepath)
        logging.info(f'Saved rand init model at '
                     f'{rand_init_model_out_filepath}')

        rand_em_acc = best_acc(y_true, y_rand_em)
    except:
        rand_em_acc = 0.

    logging.info('Image counts: %s' % str(Counter(y_true)))

    logging.info('only k-means accuracy for classes %s: %0.9f'
                 % (', '.join(map(str, class_ids)), 
                    kmeans_acc))
    logging.info('k-means + em accuracy for classes %s: %0.9f'
                 % (', '.join(map(str, class_ids)), 
                    kmeans_em_acc))
    logging.info('random init + em accuracy for classes %s: %0.9f' 
                 % (', '.join(map(str, class_ids)), 
                    rand_em_acc))

    np.savez(preds_out_filepath,
             y_true=y_true, y_kmeans=y_kmeans,
             y_kmeans_em=y_kmeans_em, y_rand_em=y_rand_em)
    logging.info(f'Saved predictions at {preds_out_filepath}')


if __name__ == '__main__':
    app.run(main)
