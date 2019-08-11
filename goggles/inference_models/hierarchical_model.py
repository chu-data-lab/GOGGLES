from goggles.inference_models.semi_supervised_models import SemiGMM,SemiBMM
import numpy as np
from tqdm import tqdm
import random
from goggles.affinity_matrix_construction.construct import construct_affinity_matrices


def infer_labels(affinity_matrix_list, dev_set_indices, dev_set_labels,seed=0,evaluate=True):
    """
    infer labels by a hierarchical inference model
    :param affinity_matrix_list:
    :param dev_set_indices: list of indices of the images in the development set
    :param dev_set_labels: list of labels of the images in the development set
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    n_classes = len(set(dev_set_labels))
    LPs = []
    for af_matrix in tqdm(affinity_matrix_list):
        base_model = SemiGMM(covariance_type="diag",n_components=n_classes)
        lp = base_model.fit_predict(af_matrix,dev_set_indices,dev_set_labels)
        LPs.append(lp)
    LPs_array = np.hstack(LPs)
    ensemble_model = SemiBMM(n_components=n_classes)
    predicted_labels = ensemble_model.fit_predict(LPs_array,dev_set_indices,dev_set_labels,evaluate)
    return predicted_labels


def generate_labels(path_to_images,dev_set_indices, dev_set_labels, seed = 0,evaluate = True):
    """
    generate labels for images
    :param path_to_images: path to the folder where images need to be labeled
    :param dev_set_indices: list of indices of the images in the development set
    :param dev_set_labels: list of labels of the images in the development set
    :return: probability of each image (totaling N) belonging to each class (totaling K),
         a N X K numpy array
    """
    random.seed(seed)
    np.random.seed(seed)
    affinity_matrix_list = construct_affinity_matrices(path_to_images)
    predicted_labels = infer_labels(affinity_matrix_list, dev_set_indices, dev_set_labels,seed,evaluate)
    return predicted_labels


if __name__ == "__main__":
    labels = generate_labels("",[1,2,3,4,5,6,7,8,9,10],[0,0,0,0,0,1,1,1,1,1],seed=0)