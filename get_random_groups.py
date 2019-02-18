from absl import app, flags
import numpy as np

from goggles.opts import DATA_DIR_MAP
from goggles.data.awa2.metadata import load_awa2_metadata
from goggles.data.cub.metadata import load_cub_metadata

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'seed', 42,
    'Random Seed')
flags.DEFINE_enum(
    'dataset', None, ['awa2', 'cub'],
    'Dataset for analysis')
flags.DEFINE_integer(
    'num_groups', 10,
    'Number of groups')
flags.DEFINE_integer(
    'group_size', 2,
    'Size of the group')

flags.mark_flag_as_required('dataset')


def main(argv):
    del argv  # unused

    np.random.seed(FLAGS.seed)

    load_fn = {
        'cub': load_cub_metadata,
        'awa2': load_awa2_metadata
    }[FLAGS.dataset]
    data_dir = DATA_DIR_MAP[FLAGS.dataset]

    list_species = load_fn(data_dir)[0]
    list_species = map(lambda x: x.id, list_species)

    groups = set()
    while True:
        group = np.random.choice(list_species,
                                 FLAGS.group_size,
                                 replace=False)

        groups.add(tuple(sorted(group)))
        if len(groups) == FLAGS.num_groups:
            break

    for group in sorted(groups):
        print(','.join(map(str, group)))


if __name__ == '__main__':
    app.run(main)
