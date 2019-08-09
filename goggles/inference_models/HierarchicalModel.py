from goggles.inference_models.SemiSupervisedBernoulliMixture import SemiBMM
from goggles.inference_models.SemiSupervisedGuassianMixture import SemiGMM
import numpy as np

class HierarchicalModel:
    """
    Goggles hierarchical inference model
    """
    def __init__(self):
        return

    def generate_labels(self, affinity_matrix_list, dev_set):
        LPs = []
        for af_matrix in affinity_matrix_list:
            base_model = SemiGMM(covariance_type="diag")
            base_model.fit(af_matrix,dev_set)
            lp = np.array(base_model.predict_proba(af_matrix))
            LPs.append(lp)
        LPs_array = np.hstack(LPs)
        ensemble_model = SemiBMM()
        ensemble_model.fit(LPs_array)
        predicted_labels = ensemble_model.predict_proba(LPs_array)
        return predicted_labels

