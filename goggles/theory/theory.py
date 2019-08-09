from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt


def generate_d_matrix(n, acc, K):
    """
    generate simulated d_matrix
    :param n: number of labeled examples per class
    :param acc: accuracy
    :param K: number of classes
    :return:
    """
    d_rows = []
    for i in range(K):
        pvals = [(1 - acc) / (K - 1)] * K
        pvals[i] = 0
        row_i = np.squeeze(np.random.multinomial(int(n * (1 - acc)), pvals, size=1))
        row_i[i] = int(n * acc)
        d_rows.append(row_i)
    d = np.squeeze(np.array(d_rows))
    return d

class DevSetTheory:
    def __init__(self,d_m):
        self.d = d_m
        self.acc_list,self.p_alpha_list= self.p_alphas()

    def log_likelihood(self,acc):
        log_L = 0
        for i in range(d.shape[0]):
            for j in range(d.shape[0]):
                if i == j:
                    log_L += d[i,j]*np.log(acc)
                else:
                    log_L += d[i,j]*np.log(1-acc)
        return log_L

    def P_d(self):
        prior = beta(1,1)
        P = 0
        n=100
        for acc in np.linspace(0.01,0.99,n):
            P+=prior.pdf(acc)*np.exp(self.log_likelihood(self.d,acc))*1/n
        return P
    def p_alphas(self):
        P_d_matrix = self.P_d()
        prior = beta(1, 1)
        n = 100
        p_alpha_list = []
        acc_list = []
        for acc in np.linspace(0.01, 0.99, n):
            p = prior.pdf(acc) * np.exp(self.log_likelihood(acc)) / P_d_matrix * 1 / n
            p_alpha_list.append(p)
            acc_list.append(acc)
        return acc_list,p_alpha_list

    def p_acc_greater(self,target_acc):
        acc_greater = 0
        for i in range(len(self.acc_list)):
            acc = self.acc_list[i]
            if acc > target_acc:
                acc_greater+=self.p_alpha_list[i]
        return acc_greater

    def feasibility_test(self,epsilon = 0.7):
        """
        The probability of the task being feasible
        :param d_m: cluster-class matrix, each row corresponds to a class and each column corresponds to a cluster
        :param epsilon: threshold of the estimated accuracy that make the task considered to be feasible
        :return: the probability of feasibility
        """
        self.p_acc_greater(epsilon)
        return self.acc_greater



if __name__ == "__main__":
    d = generate_d_matrix(15,0.72,2)
    theory = DevSetTheory(d)
    acc_greater = theory.p_acc_greater(0.7)
    ps = theory.p_alpha_list
    print(acc_greater)
    plt.plot(np.linspace(0.01, 0.99, 100),ps)
    plt.show()
