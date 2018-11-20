import numpy as np


def make_labeling_matrix(model, dataset, score_threshold=0.6):
    """
    :type model: goggles.models.semantic_ae.SemanticAutoencoder
    :type dataset: goggles.data.cub.dataset.CUBDataset
    :type score_threshold: float
    """
    labeling_matrix = list()
    true_labels = list()
    scores = list()

    all_labels = dataset.get_labels()
    all_species = all_labels.values()
    all_species = list(sorted(
        all_species, key=lambda s:s.id))

    assert len(all_species) == 2  # binary labeling functions
    species_1, species_2 = all_labels[0], all_labels[1]

    all_attributes = set()
    for species in all_species:
        all_attributes = all_attributes.union(species.attributes)

    common_attributes = set(attr for attr in all_attributes)
    for species in all_species:
        common_attributes = common_attributes.intersection(species.attributes)

    for image, image_label, _, _ in dataset:
        assert image_label in [0, 1]
        true_labels.append(-1 if image_label == 0 else 1)

        labeling_functions = list()
        prototype_scores = model.predict_prototype_scores(image)
        for attribute in sorted(all_attributes, key=lambda a: a.id):
            attribute_label = dataset.get_attribute_label(attribute)

            lf = None
            if attribute not in common_attributes:
                if attribute in species_1.attributes:
                    lf = -1 if prototype_scores[attribute_label] >= score_threshold else 0
                else:
                    lf = 1 if prototype_scores[attribute_label] >= score_threshold else 0
            else:
                lf = 0

            assert lf is not None
            labeling_functions.append(lf)

            prototype_scores = [v for k, v in sorted(prototype_scores.items(), key=lambda x: x[0])]
            scores.append(prototype_scores)

        labeling_matrix.append(labeling_functions)

    scores = np.array(scores)
    labeling_matrix = np.array(labeling_matrix)

    return labeling_matrix, scores, true_labels

