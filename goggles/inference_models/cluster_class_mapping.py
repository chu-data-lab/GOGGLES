import numpy as np
from scipy.optimize import linear_sum_assignment
from goggles.theory.theory import DevSetTheory


def construct_D(y_cluster,y_class):
    """
    construct D_matrix from cluster labels and class labels
    :param y_cluster: 1-d numpy array of cluster ids
    :param y_class: 1-d numpy array of class ids
    :return: 2-d numpy array, D_matrix[i,j] is the number of instances
    in the ith class that falls into the jth cluster
    """
    K = len(set(y_class))
    D = np.zeros(shape=(K,K))
    for i in range(K):
        for j in range(K):
            D[i][j] = np.sum((y_cluster==i)&(y_class==j))
    D_matrix = D.T
    return D_matrix


def solve_mapping(y_cluster,y_class,evaluate=False):
    """
    obtain class to cluster mapping
    :param y_cluster: 1-d numpy array of cluster ids
    :param y_class: 1-d numpy array of class ids
    :return: a list of integers where the ith element is the corresponding cluster for the ith class
    """
    D = construct_D(y_cluster,y_class)
    row_ind, col_ind = linear_sum_assignment(-D)
    if evaluate:
        #print(D)
        theory = DevSetTheory(d_matrix=D)
        print("The feasibility probability is:", theory.feasibility_test())
        print("The probability of the dev set being sufficient is:", theory.dev_set_sufficiency_test())
    return col_ind