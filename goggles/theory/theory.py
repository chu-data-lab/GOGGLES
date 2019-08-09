from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
def acc_distri(n,acc,p=0.9):
    n_s, n_f = int(n*acc), int(n*(1-acc))
    b = beta(2+n_s,1+n_f)
    eta = n_s/(n_s+n_f)
    return n_s+ n_f ,eta,1-b.cdf(0.7)

print(acc_distri(20,0.8))

def log_L_K2(d,acc):
    log_L = 0
    for i in range(d.shape[0]):
        for j in range(d.shape[0]):
            if i == j:
                log_L += d[i,j]*np.log(acc)
            else:
                log_L += d[i,j]*np.log(1-acc)
    return log_L

def log_L_K2_max(d,acc):
    log_L = log_L_K2(d,acc)
    return log_L
    #log_L_ = log_L_K2(d[::-1,:],acc)
    #return max(log_L,log_L_)


def P_d(d_matrix):
    prior = beta(1,1)
    P = 0
    n=100
    for acc in np.linspace(0.01,0.99,n):
        P+=prior.pdf(acc)*np.exp(log_L_K2_max(d_matrix,acc))*1/n
    return P

def p_acc(d_matrix,acc):
    P_d_matrix = P_d(d_matrix)
    prior = beta(1, 1)
    return prior.pdf(acc)*np.exp(log_L_K2_max(d_matrix,acc))/P_d_matrix

def p_acc_greater(d_matrix,target_acc):
    acc_all = 0
    acc_greater = 0
    P_d_matrix = P_d(d_matrix)
    prior = beta(1, 1)
    n=100
    ps = []
    for acc in np.linspace(0.01, 0.99, n):
        p = prior.pdf(acc) * np.exp(log_L_K2_max(d_matrix, acc)) / P_d_matrix*1/n
        ps.append(p)
        acc_all+=p
        if acc > target_acc:
            acc_greater+=p
    return acc_all,acc_greater,ps

def d_matrix(n, acc, K):
    d_rows = []
    for i in range(K):
        pvals = [(1-acc)/(K-1)]*K
        pvals[i] = 0
        row_i = np.squeeze(np.random.multinomial(int(n*(1-acc)),pvals,size=1))
        row_i[i] = int(n*acc)#n-np.sum(row_i)
        d_rows.append(row_i)
    d = np.squeeze(np.array(d_rows))
    print("acc",np.sum(d),np.trace(d)/np.sum(d))
    return d

d = d_matrix(15,0.72,2)
acc_all,acc_greater,ps = p_acc_greater(d,0.7)
print(acc_all,acc_greater)
plt.plot(np.linspace(0.01, 0.99, 100),ps)
plt.show()
