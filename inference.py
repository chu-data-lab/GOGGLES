from collections import Counter
import glob
import os
from types import SimpleNamespace

from absl import app, flags, logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode, norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, completeness_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm, trange

from goggles.constants import *
from goggles.data.awa2.dataset import AwA2Dataset
from goggles.data.cub.dataset import CUBDataset


FLAGS = flags.FLAGS

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

        def log_cdf(self, s):
            return norm.logcdf(s + 1e-5000, loc=self.mu, scale=self.std)

        def log_sf(self, s):
            return norm.logsf(s + 1e-5000, loc=self.mu, scale=self.std)
    
    def __init__(self, scores, cols, y):
        self._scores = scores
        self._cols = cols
        self._y = y
        
        self._num_rows = scores.shape[0]
        self._num_cols = scores.shape[1]
        self._labels = list(sorted(np.unique(y)))
        self.p1 = Counter(list(y))[1] / float(len(y))
        
        assert len(y) == self._num_rows
        
        self._params = list()
        for j in range(self._num_cols):
            self._params.append(self.fit_conditional_parameters(j))
        
    def y(self, i):
        return self._y[i]
    
    def z(self, j):
        i, _ = self._cols[j]
        return self.y(i)
    
    def s(self, i, j):
        return self._scores[i, j]
    
    def log_alpha(self, j, s):
        gaussian = self._params[j][1]
        
        if self.z(j) == 1:
            return gaussian.log_cdf(s)
        elif self.z(j) == 0:
            return gaussian.log_sf(s)
        else:
            raise ValueError('Only binary labels supported')
            
    def log_beta(self, j, s):
        gaussian = self._params[j][0]
        
        if self.z(j) == 0:
            return gaussian.log_cdf(s)
        elif self.z(j) == 1:
            return gaussian.log_sf(s)
        else:
            raise ValueError('Only binary labels supported')
            
    def log_a(self, i):
        log_ai = 0.
        
        for j in range(self._num_cols):
            sij = self.s(i, j)
            log_ai += self.log_alpha(j, sij)
                
        return log_ai
    
    def log_b(self, i):
        log_bi = 0.
        
        for j in range(self._num_cols):
            sij = self.s(i, j)
            log_bi += self.log_beta(j, sij)
                
        return log_bi
    
    def tau(self, i):
        log_ai = self.log_a(i)
        log_bi = self.log_b(i)
        log_p1 = np.log(self.p1)
        
        ai = np.exp(log_ai)
        bi = np.exp(log_bi)
        p1 = self.p1
        
        t = np.exp(log_ai + log_p1 - np.log((ai * p1) + (bi * (1 - p1)) + 1e-5000))
        return t if t <= 1. else 1.
    
    def update_model(self, y):
        self.__init__(self._scores, self._cols, y)
            
    def get_class_wise_scores(self, j):
        class_wise_scores = dict()
        for label in self._labels:
            class_wise_scores[label] = \
                self._scores[np.where(self._y == label), j].flatten()
    
        return  class_wise_scores
    
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
    
    @classmethod
    def run_em(cls, scores, cols, y_init, max_iter=100):
        n = y_init.shape[0]
        y = np.array(y_init)
        
        model = cls(scores, cols, y)
        with tqdm(range(max_iter), leave=True) as pbar:
            for _ in pbar:
                # E-step
                y_new = list()
                for i in trange(n, leave=True):
                    tau_i = model.tau(i)
                    y_i = np.random.choice(2, 1, p=[1 - tau_i, tau_i])[0]
                    y_new.append(y_i)

                y_new = np.array(y_new)

                if np.linalg.norm(y_new - y) == 0:
                    break

                # M-step
                y = np.array(y_new)
                model.update_model(y)
            
        return y_new



def main(argv):
    del argv  # unused
    
    class_ids = list(map(int, FLAGS.class_ids))
    
    logging.info('calculating accuracies for classes %s from %s'
                 % (', '.join(map(str, class_ids)), FLAGS.dataset))

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
            f'vgg16_layer30-{FLAGS.dataset}-%d_%d-scores.npz' 
            % tuple(class_ids)))

    y_pred = KMeans(n_clusters=2).fit_predict(scores)
    kmeans_acc = best_acc(y_true, y_pred)
    
    try:
        y_em_kmeans = GogglesProbabilisticModel.run_em(scores, col_ids, y_pred)
        em_kmeans_acc = best_acc(y_true, y_em_kmeans)
    except:
        em_kmeans_acc = 0.
    
    
    try:
        np.random.seed(sum(v * (10**(3*i)) for i,v in enumerate(class_ids)))
        y_init = np.random.randint(2, size=scores.shape[0])
        y_em = GogglesProbabilisticModel.run_em(scores, col_ids, y_init)
        em_rand_acc = best_acc(y_true, y_em)
    except:
        em_rand_acc = 0.

    logging.info('Image counts: %s' % str(Counter(y_true)))

    logging.info('only kmeans accuracy for classes %s: %0.9f' 
                 % (', '.join(map(str, class_ids)), 
                    kmeans_acc))
    logging.info('kmeans + em accuracy for classes %s: %0.9f' 
                 % (', '.join(map(str, class_ids)), 
                    em_kmeans_acc))
    logging.info('random init + em accuracy for classes %s: %0.9f' 
                 % (', '.join(map(str, class_ids)), 
                    em_rand_acc))


if __name__ == '__main__':
    app.run(main)
