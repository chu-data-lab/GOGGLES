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

def obtain_dev_set():
    """
    :return: development set, a list a tuples [(index of the image, class label),...]
    """
    dev_set = [(25,0),(30,0),(51,1),(52,1)]
    return dev_set