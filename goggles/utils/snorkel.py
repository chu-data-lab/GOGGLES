import numpy as np


def make_labeling_matrix(model, dataset, score_threshold=0.6):
    """
    :type model: goggles.models.semantic_ae.SemanticAutoencoder
    :type dataset: goggles.data.cub.dataset.CUBDataset
    :type score_threshold: float
    """
    labeling_matrix = list()
    true_labels = list()

    all_labels = dataset.get_labels()
    all_species = all_labels.values()
    all_species = list(sorted(
        all_species, key=lambda s:s.id))

    assert len(all_species) == 2  # binary labeling functions
    species_1, species_2 = tuple(all_species)

    all_attributes = set()
    for species in all_species:
        all_attributes = all_attributes.union(species.attributes)

    common_attributes = set(attr for attr in all_attributes)
    for species in all_species:
        common_attributes = common_attributes.intersection(species.attributes)

    for image, image_label, attribute_labels, num_nonzero_attributes in dataset:
        true_labels.append(image_label)

        labeling_functions = list()
        prototype_scores = model.predict_prototype_scores(image)
        for attribute in sorted(all_attributes, key=lambda a: a.id):
            attribute_label = dataset.get_attribute_label(attribute)

            lf = None
            if attribute not in common_attributes:
                if attribute in species_1.attributes:
                    lf = 1 if prototype_scores[attribute_label] >= score_threshold else 0
                else:
                    lf = -1 if prototype_scores[attribute_label] >= score_threshold else 0
            else:
                lf = 0

            assert lf is not None
            labeling_functions.append(lf)

        labeling_matrix.append(labeling_functions)
        labeling_matrix = np.array(labeling_matrix)

    return labeling_matrix, true_labels

