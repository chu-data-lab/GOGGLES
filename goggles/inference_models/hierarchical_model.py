from .semi_supervised_models import SemiGMM,SemiBMM
import numpy as np
from tqdm import tqdm


def generate_labels(affinity_matrix_list, dev_set_indices,dev_set_labels):
    """
    Generate labels by a hierarchical inference model
    :param affinity_matrix_list:
    :param dev_set:
    :return:
    """
    n_classes = len(set(dev_set_labels))
    LPs = []
    for af_matrix in tqdm(affinity_matrix_list):
        base_model = SemiGMM(covariance_type="diag",n_components=n_classes)
        base_model.fit(af_matrix,dev_set_indices,dev_set_labels)
        lp = np.array(base_model.predict_proba(af_matrix))
        LPs.append(lp)
    LPs_array = np.hstack(LPs)
    ensemble_model = SemiBMM(n_components=n_classes)
    predicted_labels = ensemble_model.fit_predict(LPs_array,dev_set_indices,dev_set_labels)
    return predicted_labels

