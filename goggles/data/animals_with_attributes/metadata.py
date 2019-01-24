import os
import glob

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
    def __init__(self, id_, name, certainty):
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

    datum_by_id = dict()
    datum_folder = os.path.join(animal_data_dir, 'JPEGImages')
    for i, image in enumerate(glob.glob(datum_folder)):
        print i, image
        if i == 5:
            break



    # all_species, all_attributes, all_images_data = \
    #     list(sorted(species_by_id.values(), key=lambda s: s.id)), \
    #     list(sorted(attributes_by_id.values(), key=lambda a: a.id)), \
    #     list(sorted(datum_by_id.values(), key=lambda d: d.id))
    #
    # return all_species, all_attributes, all_images_data

if __name__ == '__main__':
    from goggles.constants import *

    print(load_animals_metadata(ANIMALS_DIR))
