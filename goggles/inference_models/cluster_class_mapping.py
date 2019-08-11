import numpy as np
from scipy.optimize import linear_sum_assignment
from goggles.theory.theory import DevSetTheory

def construct_D(y_cluster,y_class):
    K = len(set(y_class))
    D = np.zeros(shape=(K,K))
    for i in range(K):
        for j in range(K):
            D[i][j] = np.sum((y_cluster==i)&(y_class==j))
    return D.T


def solve_mapping(y_cluster,y_class,evaluate=False):
    """
    obtain class to cluster mapping
    :param y_cluster:
    :param y_class:
    :return: a list of integers where the ith element is the corresponding cluster for the ith class
    """
    D = construct_D(y_cluster,y_class)
    row_ind, col_ind = linear_sum_assignment(-D)
    if evaluate:
        print(D)
        theory = DevSetTheory(d_matrix=D)
        print("The feasible probability is:", theory.feasibility_test())
        print("The probability that the dev set is enough is", theory.dev_set_sufficiency_test())
    return col_ind