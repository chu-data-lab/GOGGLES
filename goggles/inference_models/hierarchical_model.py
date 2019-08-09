from goggles.inference_models.semi_supervised_bernoulli_mixture import SemiBMM
from goggles.inference_models.semi_supervised_guassian_mixture import SemiGMM
import numpy as np
from tqdm import tqdm


def generate_labels(affinity_matrix_list, dev_set):
    """
    Generate labels by a hierarchical inference model
    :param affinity_matrix_list:
    :param dev_set:
    :return:
    """
    LPs = []
    for af_matrix in tqdm(affinity_matrix_list):
        base_model = SemiGMM(covariance_type="diag")
        base_model.fit(af_matrix,dev_set)
        lp = np.array(base_model.predict_proba(af_matrix))
        LPs.append(lp)
    LPs_array = np.hstack(LPs)
    ensemble_model = SemiBMM()
    ensemble_model.fit(LPs_array)
    predicted_labels = ensemble_model.predict_proba(LPs_array)
    return predicted_labels

