import os


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

        self.certainty = certainty


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


def load_cub_metadata(cub_data_dir):
    attributes_by_id = dict()
    attributes_file = os.path.join(cub_data_dir, 'attributes.txt')
    with open(attributes_file, 'r') as f:
        for l in f.readlines():
            id_, name = l.strip().split()
            id_ = int(id_)
            attributes_by_id[id_] = Attribute(id_, name)

    species_by_id = dict()
    species_file = os.path.join(cub_data_dir, 'CUB_200_2011', 'classes.txt')
    with open(species_file, 'r') as f:
        for l in f.readlines():
            id_, name = l.strip().split()
            id_ = int(id_)
            species_by_id[id_] = Species(id_, name)

    datum_by_id = dict()
    datum_file = os.path.join(cub_data_dir, 'CUB_200_2011', 'images.txt')
    with open(datum_file, 'r') as f:
        for l in f.readlines():
            id_, path = l.strip().split()
            id_ = int(id_)
            species_name = path.split('/')[0]
            species_id = int(species_name.split('.')[0])
            species = species_by_id[species_id]
            datum_by_id[id_] = Datum(id_, path, species)

    split_file = os.path.join(cub_data_dir, 'CUB_200_2011', 'train_test_split.txt')
    with open(split_file, 'r') as f:
        for l in f.readlines():
            id_, is_for_training = l.strip().split()
            id_ = int(id_)
            is_for_training = int(is_for_training) == 1
            datum_by_id[id_].set_for_training(is_for_training)

    higher_order_attributes = set(a.higher_order_name for a in attributes_by_id.values())
    higher_order_attribute_ids = {
        ha: [i for i, a in attributes_by_id.items() if ha in str(a)]
        for ha in higher_order_attributes}
    species_attributes_file = os.path.join(
        cub_data_dir, 'CUB_200_2011', 'attributes', 'class_attribute_labels_continuous.txt')
    with open(species_attributes_file) as f:
        for i, l in enumerate(f.readlines()):
            species_id = i + 1
            species_attributes_scores = list(map(float, l.strip().split()))
            for ha, a_ids in higher_order_attribute_ids.items():
                for a_id in a_ids:
                    score = species_attributes_scores[a_id - 1]
                    if score > 50:
                        species_by_id[species_id].add_attribute(attributes_by_id[a_id])

    certainties_by_id = {1: 'not visible', 2: 'guessing', 3: 'probably', 4: 'definitely'}
    image_attributes_file = os.path.join(cub_data_dir, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt')
    with open(image_attributes_file, 'r') as f:
        for l in f.readlines():
            datum_id, attr_id, is_present, certainty = tuple(map(int, l.strip().split()[:4]))
            if is_present == 1:
                attr = AttributeAnnotation(attr_id,
                                           attributes_by_id[attr_id].name,
                                           certainties_by_id[certainty])
                datum_by_id[datum_id].add_attribute_annotation(attr)

    all_species, all_attributes, all_images_data = \
        list(sorted(species_by_id.values(), key=lambda s: s.id)), \
        list(sorted(attributes_by_id.values(), key=lambda a: a.id)), \
        list(sorted(datum_by_id.values(), key=lambda d: d.id))

    return all_species, all_attributes, all_images_data

if __name__ == '__main__':
    from goggles.constants import *

    print(load_cub_metadata(CUB_DATA_DIR))
