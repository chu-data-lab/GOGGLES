from goggles import construct_image_affinity_matrices, GogglesDataset,infer_labels
import numpy as np
from sklearn.metrics import accuracy_score
if __name__ == '__main__':
    dataset = GogglesDataset.load_all_data("data/cub_dataset")
    afs = construct_image_affinity_matrices(dataset)
    dev_set_indices, dev_set_labels = [0,1,2,90,91,92],[0,0,0,1,1,1]
    y_true = np.ones(len(dataset.images_filename_list))
    y_true[:int(y_true.shape[0]/2)] = 0
    prob = infer_labels(afs,dev_set_indices,dev_set_labels)
    pred_labels = np.argmax(prob,axis=1).astype(int)
    print("accuracy", accuracy_score(y_true,pred_labels))
