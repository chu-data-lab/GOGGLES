from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from goggles.inference_models.cluster_class_mapping import solve_mapping
import numpy as np
DEL = 1e-300

def update_prob_using_mapping(prob, dev_set_indices, dev_set_labels,evaluate=False):
    cluster_labels = np.argmax(prob, axis=1)
    dev_cluster_labels = cluster_labels[dev_set_indices]
    cluster_class_mapping = solve_mapping(dev_cluster_labels, dev_set_labels,evaluate)
    prob = prob[:, cluster_class_mapping]
    return prob


def set_prob_dev_values(prob, dev_set_indices, dev_set_labels):
    #prob[dev_set_indices, :] = 0
    #for i in range(len(dev_set_indices)):
    #    prob[dev_set_indices[i], dev_set_labels[i]] = 1
    return prob



def pmf_bernoulli(s,mu):
    return np.exp(np.sum(s*np.log(mu+DEL)+(1-s)*np.log(1-mu+DEL),axis=1))


class ConvergenceMeter:
    def __init__(self, num_converged, rate_threshold,
                 diff_fn=lambda a, b: abs(a - b)):
        self._num_converged = num_converged
        self._rate_threshold = rate_threshold
        self._diff_fn = diff_fn
        self._diff_history = list()
        self._last_val = None

    def offer(self, val):
        if self._last_val is not None:
            self._diff_history.append(
                self._diff_fn(val, self._last_val))

        self._last_val = val

    @property
    def is_converged(self):
        if len(self._diff_history) < self._num_converged:
            return False

        return np.mean(
            self._diff_history[-self._num_converged:]) \
               <= self._rate_threshold

class SemiGMM(GaussianMixture):
    """
    Goggles Semi-supervised Guassian Mixture model adapted from scikit-learn.
    The cluster-to-class mapping is performed based section 4.3
    """

    def __init__(self, n_components=1, covariance_type='full', tol=1e-4, reg_covar=1e-6):
        super().__init__(
            n_components=n_components, covariance_type = covariance_type,tol=tol,reg_covar=reg_covar)

    def fit(self, X, dev_set_indices,dev_set_labels):
        self.dev_set_indices = np.array(dev_set_indices)
        self.dev_set_labels = np.array(dev_set_labels)
        return super(SemiGMM, self).fit(X)

    def fit_predict(self,X, dev_set_indices,dev_set_labels):
        self.fit(X, dev_set_indices,dev_set_labels)
        return self.predict_proba(X)

    def _estimate_log_prob_resp(self,X):
        log_prob_norm, log_resp = super()._estimate_log_prob_resp(X)
        prob = np.exp(log_resp)
        prob = update_prob_using_mapping(prob, self.dev_set_indices, self.dev_set_labels)
        prob = set_prob_dev_values(prob, self.dev_set_indices, self.dev_set_labels)
        log_resp = np.log(prob + DEL)
        return log_prob_norm, log_resp


class SemiBMM:
    def __init__(self,n_components):
        self.K = n_components
        self.pi = np.ones(self.K)*1/self.K
        self.mu = np.zeros(self.K)


    def initalization(self,X):
        km = KMeans(n_clusters=self.K)
        y_init = km.fit_predict(X)
        prob = np.zeros(shape=(X.shape[0],self.K))
        for i in range(X.shape[0]):
            prob[i,y_init[i]] = 1
        return prob


    def fit_predict(self,X, dev_set_indices,dev_set_labels,evaluate):
        prob = self.initalization(X)
        self.dev_set_indices = np.array(dev_set_indices)
        self.dev_set_labels = np.array(dev_set_labels)
        convergence = ConvergenceMeter(20, 1e-6, diff_fn=lambda a, b: np.linalg.norm(a - b))
        while not convergence.is_converged:
            self.M_step(X,prob)
            prob = self.E_step(X)
            convergence.offer(prob)
        if evaluate:
            prob = self.E_step(X,evaluate)
        return prob


    def E_step(self,X,evaluate=False):
        prob = np.zeros(shape=(X.shape[0],self.K))
        pi_mul_P = []
        for i in range(self.K):
            pi_mul_P.append(self.pi[i]*pmf_bernoulli(X,self.mu[i]))
        pi_mul_P_sum = np.sum(pi_mul_P,axis=0)
        for i in range(self.K):
            prob[:,i] = pi_mul_P[i]/pi_mul_P_sum

        prob = update_prob_using_mapping(prob, self.dev_set_indices, self.dev_set_labels,evaluate)
        prob = set_prob_dev_values(prob, self.dev_set_indices, self.dev_set_labels)
        return prob


    def M_step(self,X, prob):
        Ns = np.sum(prob,axis=0)
        self.pi = Ns/prob.shape[1]
        self.mu = [np.sum((X.T*prob[:,i]).T, axis=0)/Ns[i] for i in range(self.K)]

if __name__ == "__main__":
    data = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    bmm = SemiBMM(2)
    print(bmm.fit_predict(np.array(data),dev_set_indices=[0,2],dev_set_labels=[0,1]))