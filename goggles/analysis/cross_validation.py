from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


def get_best_thresholds(model, dataset):
    """
    :type model: goggles.models.semantic_ae.SemanticAutoencoder
    :type dataset: goggles.data.cub.dataset.CUBDataset
    """

    true_labels = list()
    prototype_scores = defaultdict(list)

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

        ps = model.predict_prototype_scores(image)
        for attribute in sorted(all_attributes, key=lambda a: a.id):
            attribute_label = dataset.get_attribute_label(attribute)
            prototype_scores[attribute_label].append(ps[attribute_label])

    thresholds = dict()
    for attribute in sorted(all_attributes, key=lambda a: a.id):
        attribute_label = dataset.get_attribute_label(attribute)

        desired_label = None
        if attribute not in common_attributes:
            desired_label = -1 if attribute in species_1.attributes else 1


        if desired_label is not None:
            undesired_label = 1 if desired_label == -1 else -1

            scores = prototype_scores[attribute_label]
            min_score, max_score = min(scores), max(scores)

            best_performance = -1.
            for threshold in np.linspace(min_score, max_score, 100):
                predicted_labels = [desired_label if score >= threshold
                                    else undesired_label
                                    for score in scores]

                performance = f1_score(true_labels, predicted_labels,
                                       pos_label=desired_label)

                if performance > best_performance:
                    thresholds[attribute_label] = threshold
                    best_performance = performance
        else:
            thresholds[attribute_label] = 1.

    return thresholds
