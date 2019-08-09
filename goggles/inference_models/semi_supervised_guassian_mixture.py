from sklearn.mixture import GaussianMixture
import numpy as np
DEL = 1e-200


class SemiGMM(GaussianMixture):
    """
    Goggles Semi-supervised Guassian Mixture model adapted from scikit-learn.
    The cluster-to-class mapping is performed based section 4.3
    """

    def __init__(self, n_components=1, covariance_type='full', tol=1e-4, reg_covar=1e-6):
        super().__init__(
            n_components=n_components, covariance_type = covariance_type,tol=tol,reg_covar=reg_covar)

    def fit(self, X, dev_set):
        self.dev_set = dev_set
        return super(SemiGMM, self).fit(X)

    def _estimate_log_prob_resp(self,X):
        log_prob_norm, log_resp = super()._estimate_log_prob_resp(X)
        prob = np.exp(log_resp)
        if len(self.dev_set[0]) > 0:
            majority_0 = np.mean(prob[:, 1][self.dev_set[0]])
            majority_1 = np.mean(prob[:, 1][self.dev_set[1]])
            if majority_1 < majority_0:
                prob[:, [0, 1]] = prob[:, [1, 0]]
            prob[:, 1][self.dev_set[1]] = 1
            prob[:, 0][self.dev_set[1]] = 0
            prob[:, 1][self.dev_set[0]] = 0
            prob[:, 0][self.dev_set[0]] = 1
        log_resp = np.log(prob + DEL)
        return log_prob_norm, log_resp