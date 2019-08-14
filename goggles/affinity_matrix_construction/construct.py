import numpy as np
from goggles.affinity_matrix_construction.image_AF.neural_network_AFs import nn_AFs


def construct_image_affinity_matrices(dataset,cache=True):
    """
    :param GogglesDataset instance
    :return: a list of affinity matrices
    """
    matrix_list = []
    for layer_idx in [4,9,16,23,30]:#all max pooling layers
        matrix_list.extend(nn_AFs(dataset,layer_idx,10,cache))
    return matrix_list
