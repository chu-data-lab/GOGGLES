import numpy as np

def construct_affinity_matrices(path_to_image_foler=""):
    """
    :param path_to_image_foler: path to the folder where images need to be labeled
    :return: a list of affinity matrices
    """
    matrix_list = []
    for i in range(10):
        matrix_list.append(np.random.rand(100,100))
    return matrix_list
