from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from .cluster_class_mapping import solve_mapping
import numpy as np


def update_prob_using_dev_set(prob, dev_set_indices,dev_set_labels):
    cluster_labels = np.argmax(prob, axis=1)
    dev_cluster_labels = cluster_labels[dev_set_indices]
    cluster_class_mapping = solve_mapping(dev_cluster_labels, dev_set_labels)
    prob = prob[:, cluster_class_mapping]
    prob[dev_set_indices,:] = 0
    for i in range(len(dev_set_indices)):
        prob[dev_set_indices[i], dev_set_labels[i]] = 1
    return prob

class SemiGMM(GaussianMixture):
    """
    Goggles Semi-supervised Guassian Mixture model adapted from scikit-learn.
    The cluster-to-class mapping is performed based section 4.3
    """

    def __init__(self, n_components=1, covariance_type='full', tol=1e-4, reg_covar=1e-6):
        super().__init__(
            n_components=n_components, covariance_type = covariance_type,tol=tol,reg_covar=reg_covar)

    def fit(self, X, dev_set_indices,dev_set_labels):
        self.dev_set_indices = dev_set_indices
        self.dev_set_labels = dev_set_labels
        return super(SemiGMM, self).fit(X)

    def _estimate_log_prob_resp(self,X):
        log_prob_norm, log_resp = super()._estimate_log_prob_resp(X)
        prob = np.exp(log_resp)
        prob = update_prob_using_dev_set(prob,self.dev_set_indices,self.dev_set_labels)
        log_resp = np.log(prob + 1e-200)
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
            prob[i:,y_init[i]] = 1
        return prob

    def fit(self,X, dev_set_indices,dev_set_labels):
        prob = self.initalization(X)
        self.M_step(X,prob)


    def E_step(self):
        for n in range(self.N):
            sumz = 0
            for k in range(self.K):
                self.z[n][k] = self.pk(self.data[n], k)
                sumz += self.z[n][k]
            for k in range(self.K):
                self.z[n][k] /= sumz
        prob = np.array(self.z)
        prob = update_prob_using_dev_set(prob,)

        for n in range(self.N):
            for k in range(self.K):
                self.z[n][k] = prob[n,k]

    def M_step(self,X, prob):
        Ns = np.sum(prob,axis=0)
        self.pi = Ns/prob.shape[1]
        self.mu = [np.sum(X*prob[:,i],axis=0)/Ns[i] for i in range(self.K)]


class SemiBMM_:
    """
    Semi-supervised Bernoulli Mixture model
    The cluster-to-class mapping is performed based section 4.3
    """
    def __init__(self, data, K):
        self.data = data

        self.N = len(data)
        self.D = data.shape[1]
        self.K = K

        km = KMeans(n_clusters=K)
        y_init = km.fit_predict(data)
        # model parameters
        # random initialization
        self.z = [[1 if int(y_init[n])==k else 0 for k in range(self.K)] for n in range(self.N)]
        newpi, newmu = self.M_step()
        self.pi = newpi#[1 / self.K for k in range(self.K)]
        self.mu = newmu#[[random.random() for d in range(self.D)] for k in range(self.K)]
        self.label_set = [[], [[]]]
        self.semi_examples = [np.array([]),np.array([])]
        # update parameters
        #self.learn()

    # probability of x for the k component
    def pk(self, x, k):
        resp = 1
        for i in range(len(x)):
            if x[i] == 1:
                resp *= self.mu[k][i]
            else:
                resp *= (1 - self.mu[k][i])
        return resp * self.pi[k]

    # e-m algorithm to learn the parameters
    def E_step(self):
        for n in range(self.N):
            sumz = 0
            for k in range(self.K):
                self.z[n][k] = self.pk(self.data[n], k)
                sumz += self.z[n][k]
            for k in range(self.K):
                self.z[n][k] /= sumz
        prob = np.array(self.z)
        prob = update_prob_using_dev_set(prob,)

        for n in range(self.N):
            for k in range(self.K):
                self.z[n][k] = prob[n,k]
        #print(len(self.z))




    def M_step(self):
        # M step
        N_m = [0 for k in range(self.K)]
        z_x = [[0 for d in range(self.D)] for k in range(self.K)]
        newpi = [1 / self.K for k in range(self.K)]
        newmu = [[1 / self.D for d in range(self.D)] for k in range(self.K)]
        for k in range(self.K):
            for n in range(self.N):
                N_m[k] += self.z[n][k]
                for d in range(self.D):
                    z_x[k][d] += self.z[n][k] * self.data[n][d]
            for d in range(self.D):
                newmu[k][d] = z_x[k][d] / N_m[k]
            newpi[k] = N_m[k] / self.N
        return newpi,newmu

    def learn(self, data=None):
        change = True
        niter = 0
        while change:
            change = False
            # E step
            self.E_step()
            newpi, newmu = self.M_step()

            for k in range(self.K):
                if self.pi[k] != newpi[k]:
                    change = True
                    self.pi[k] = newpi[k]
                for d in range(self.D):
                    if self.mu[k][d] != newmu[k][d]:
                        change = True
                        self.mu[k][d] = newmu[k][d]
            niter += 1
            if niter >= 100:
                break

    def printModel(self):
        print("Mu: " + str(self.mu))
        print("Pi: " + str(self.pi))

    def predict(self, x,dev_set_indices,dev_set_labels):
        resp = []
        for k in range(self.K):
            resp.append(self.pk(x, k))
        return resp
    def fit(self,x,semi_examples=None):
        self.semi_examples = semi_examples
        if semi_examples is not None:
            self.label_set[0] = set(semi_examples[0].tolist())
            self.label_set[1] = set(semi_examples[1].tolist())
        self.learn()
        return
    def predict_proba(self,data):
        return np.array(self.z)

    def predictAll(self, data):
        resp = []
        for dat in data:
            resp.append(self.predict(dat))
        return np.array(resp)

