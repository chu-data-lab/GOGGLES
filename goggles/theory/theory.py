from scipy.stats import beta
import numpy as np
import math


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
    print(np.trace(d)/np.sum(d))
    return d


def n_given_sum(n_class,n_sum,n_max,x_0):
    n_given_sum_list = [[None for _ in range(n_sum+1)] for _ in range(n_class+1)]
    def _n_given_sum_(n_class,n_sum,n_max):
        if n_given_sum_list[n_class][n_sum] is not None:
            return n_given_sum_list[n_class][n_sum]
        if n_class == 1:
            if n_max < n_sum:
                n_given_sum_list[n_class][n_sum] = 0
                return 0
            else:
                n_given_sum_list[n_class][n_sum] = 1/math.factorial(x_0 - n_sum-1)
                return n_given_sum_list[n_class][n_sum]
        if n_class < 1 and n_sum!=0:
            return 0
        n = 0
        for i in range(min(n_max,n_sum)+1):
            n += 1/math.factorial(x_0-i-1)*_n_given_sum_(n_class-1,n_sum-i,n_max)
        n_given_sum_list[n_class][n_sum] = n
        return n
    return _n_given_sum_(n_class,n_sum,n_max)


class DevSetTheory:
    def __init__(self,d_matrix):
        self.D_matrix = d_matrix
        self.n=100
        self.alpha_list, self.p_alpha_list= self.p_alphas()
        self.d_alpha = self.alpha_list[1]-self.alpha_list[0]

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
        for acc in np.linspace(1e-6, 1-1e-6,self.n):
            P+=prior.pdf(acc)*np.exp(self.log_likelihood(acc))*( 1 / self.n)
        return P
    def p_alphas(self):
        P_d_matrix = self.P_d()
        prior = beta(1, 1)
        p_alpha_list = []
        acc_list = []
        for acc in np.linspace(1e-6, 1-1e-6, self.n):
            p = prior.pdf(acc) * np.exp(self.log_likelihood(acc)) / P_d_matrix
            p_alpha_list.append(p)
            acc_list.append(acc)
        return acc_list,p_alpha_list

    def p_acc_greater(self,target_acc):
        acc_greater = 0
        for i in range(len(self.alpha_list)):
            acc = self.alpha_list[i]
            if acc > target_acc:
                acc_greater+=self.p_alpha_list[i]*self.d_alpha
        return acc_greater

    def feasibility_test(self,epsilon = 0.7):
        """
        The probability of the task being feasible
        :param d_m: cluster-class matrix, each row corresponds to a class and each column corresponds to a cluster
        :param epsilon: threshold of the estimated accuracy that make the task considered to be feasible
        :return: the probability of feasibility
        """
        acc_greater = self.p_acc_greater(epsilon)
        acc_greater = np.clip(acc_greater, 0, 1)
        return acc_greater


    def p_one_dim(self,acc,i_dim):
        n_class = self.D_matrix.shape[0]
        dev_size = np.sum(self.D_matrix[i_dim, :])
        x_0_min = int(math.ceil((dev_size + n_class - 1) / n_class))

        def p(x_0):
            y_sum = n_class * x_0 - dev_size - (n_class - 1)

            return math.factorial(dev_size) / math.factorial(x_0) * math.exp(
                math.log(acc) * x_0 + math.log((1 - acc) / (n_class - 1)) * (dev_size - x_0)) * \
                   n_given_sum(n_class - 1, y_sum, x_0 - 1, x_0)

        prob = 0
        for x_0 in range(x_0_min, dev_size + 1):
            prob += p(x_0)
        return prob

    def dev_set_sufficiency_test(self):
        """
        test whether the current dev set is sufficient.
        This offers a guidance on whether you should use a larger dev set.
        :return: the lower-bound probability whether the dev set is sufficiency enough
         to obtain the correct class-cluster mapping
        """
        p = 0
        for i in range(len(self.alpha_list)):
            alpha = self.alpha_list[i]
            p_alpha = self.p_alpha_list[i]
            pl = 1
            for i_dim in range(self.D_matrix.shape[0]):
                pl*=self.p_one_dim(alpha, i_dim)
            p = p + pl*p_alpha*self.d_alpha
        p = np.clip(p,0,1)
        return p


if __name__ == "__main__":
    d = generate_d_matrix(10,0.77,2)
    theory = DevSetTheory(d)
    acc_greater = theory.p_acc_greater(0.7)
    ps = theory.p_alpha_list
    print(acc_greater)
    print(theory.dev_set_sufficiency_test())
    #plt.plot(np.linspace(0.01, 0.99, 100),ps)
    #plt.show()
