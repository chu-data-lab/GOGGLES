import os
import glob
from collections import defaultdict
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

    @property
    def higher_order_name(self):
        return self.name.split('::')[0]


class AttributeAnnotation(Attribute):
    def __init__(self, id_, name, certainty=1):
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
        self.name = name

    def add_attribute(self, attribute):
        assert isinstance(attribute, Attribute)

        self.attributes.add(attribute)


def load_animals_metadata(animal_data_dir):
    attributes_by_id = dict()
    attributes_file = os.path.join(animal_data_dir, 'predicates.txt')
    with open(attributes_file, 'r') as f:
        for l in f.readlines():
            id_, name = l.strip().split()
            id_ = int(id_)
            attributes_by_id[id_] = Attribute(id_, name)

    species_by_id = dict()
    species_file = os.path.join(animal_data_dir, 'classes.txt')
    with open(species_file, 'r') as f:
        for l in f.readlines():
            id_, name = l.strip().split()
            id_ = int(id_)
            species_by_id[id_] = Species(id_, name)

    predicate_matrix = np.loadtxt(os.path.join(animal_data_dir, 'predicate-matrix-binary.txt'))
    attributes_by_species_id = defaultdict(list)
    num_species = 50
    num_predicates = 85
    for i in range(num_species):
        for j in range(num_predicates):
            if predicate_matrix[i][j] == 1:
                attributes_by_species_id[i + 1].append(j + 1) #species and attributes are 1-indexed

    datum_by_id = dict()
    datum_folder = os.path.join(animal_data_dir, 'JPEGImages', '*.jpg')
    id_ = 0
    for species_id in species_by_id:
        entered = False
        datum_folder = os.path.join(animal_data_dir, 'JPEGImages', species_by_id[species_id].name, '*.jpg')
        for image_path in glob.glob(datum_folder):
            id_ += 1
            print id_,  image_path
            sub_path = os.path.join(image_path.split('/')[7], image_path.split('/')[8])
            datum_by_id[id_] = Datum(id_, sub_path, species_by_id[species_id])
            for attribute_id in attributes_by_species_id[species_id]:
                attribute_annotation_obj = AttributeAnnotation(attribute_id, attributes_by_id[attribute_id].name)
                datum_by_id[id_].add_attribute_annotation(attribute_annotation_obj)
                if entered == False:
                    species_by_id[species_id].add_attribute(attributes_by_id[attribute_id])
            entered = True

    total_images = id_
    train_num = 0.7 * total_images
    for i, image_id in enumerate(datum_by_id):
        if i <= train_num:
            datum_by_id[image_id].is_for_training = True
        else:
            datum_by_id[image_id].is_for_training = False

    all_species, all_attributes, all_images_data = \
        list(sorted(species_by_id.values(), key=lambda s: s.id)), \
        list(sorted(attributes_by_id.values(), key=lambda a: a.id)), \
        list(sorted(datum_by_id.values(), key=lambda d: d.id))

    return all_species, all_attributes, all_images_data

if __name__ == '__main__':
    # from goggles.constants import *
    ANIMALS_DIR = '/media/seagate/rtorrent/AwA2/Animals_with_Attributes2'

    print(load_animals_metadata(ANIMALS_DIR))
