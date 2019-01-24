import os
import glob
import numpy as np


class _NamedClass(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, another):
        return self.__hash__() == hash(another)


class Attribute(_NamedClass):
    def __init__(self, id_, name):
        super(Attribute, self).__init__(name)

        self.id = id_


class AttributeAnnotation(Attribute):
    def __init__(self, id_, name):
        super(AttributeAnnotation, self).__init__(id_, name)


class Datum(object):
    def __init__(self, id_, path, species, attribute_annotations=None, is_for_training=False):
        if attribute_annotations is None:
            attribute_annotations = set()
        assert type(attribute_annotations) is set
        assert all(isinstance(aa, AttributeAnnotation)
                   for aa in attribute_annotations)

        self.id = id_
        self.path = path
        self.species = species
        self.attribute_annotations = attribute_annotations
        self.is_for_training = is_for_training

    def add_attribute_annotation(self, attribute_annotation):
        assert isinstance(attribute_annotation, AttributeAnnotation)

        self.attribute_annotations.add(attribute_annotation)

    def attribute_is_present(self, attribute):
        return attribute in self.attribute_annotations

    def set_for_training(self, is_for_training=True):
        self.is_for_training = is_for_training


class Species(_NamedClass):
    def __init__(self, id_, name, attributes=None):
        super(Species, self).__init__(name)

        if attributes is None:
            attributes = set()
        assert type(attributes) is set
        assert all(isinstance(a, Attribute)
                   for a in attributes)

        self.id = id_
        self.attributes = attributes

    def add_attribute(self, attribute):
        assert isinstance(attribute, Attribute)

        self.attributes.add(attribute)


def load_awa2_metadata(awa2_data_dir, train_split=0.8):
    awa2_data_dir = os.path.join(awa2_data_dir, 'Animals_with_Attributes2')

    attributes_by_id = dict()
    attributes_file = os.path.join(awa2_data_dir, 'predicates.txt')
    with open(attributes_file, 'r') as f:
        for l in f.readlines():
            id_, name = l.strip().split()
            id_ = int(id_)
            attributes_by_id[id_] = Attribute(id_, name)

    species_by_id = dict()
    species_file = os.path.join(awa2_data_dir, 'classes.txt')
    with open(species_file, 'r') as f:
        for l in f.readlines():
            id_, name = l.strip().split()
            id_ = int(id_)
            species_by_id[id_] = Species(id_, name)

    predicate_matrix = \
        np.loadtxt(os.path.join(awa2_data_dir, 'predicate-matrix-binary.txt'))
    num_species = len(species_by_id)
    num_predicates = len(attributes_by_id)
    for i in range(num_species):
        species_id = i + 1
        species = species_by_id[species_id]

        for j in range(num_predicates):
            attribute_id = j + 1
            attribute = attributes_by_id[attribute_id]

            if predicate_matrix[i][j] == 1:
                species.add_attribute(attribute)

    datum_by_id = dict()
    for species_id in species_by_id:
        species = species_by_id[species_id]
        species_data = list()
        for image_path in glob.glob(os.path.join(
                awa2_data_dir, 'JPEGImages', species.name, '*.jpg')):
            datum_id = os.path.basename(image_path)
            datum_path = os.path.join(species.name, datum_id)
            datum = Datum(datum_id, datum_path, species)

            for attribute in species.attributes:
                datum.add_attribute_annotation(
                    AttributeAnnotation(attribute.id, attribute.name))

            datum_by_id[datum_id] = datum
            species_data.append(datum)

        species_data = sorted(species_data, key=lambda d: d.id)
        num_train = int(train_split * len(species_data))
        for i in range(len(species_data)):
            is_for_training = i < num_train
            species_data[i].set_for_training(is_for_training)

    all_species, all_attributes, all_images_data = \
        list(sorted(species_by_id.values(), key=lambda s: s.id)), \
        list(sorted(attributes_by_id.values(), key=lambda a: a.id)), \
        list(sorted(datum_by_id.values(), key=lambda d: d.id))

    return all_species, all_attributes, all_images_data
