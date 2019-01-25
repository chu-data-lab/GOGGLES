from goggles.constants import *
from goggles.data.awa2.dataset import AwA2Dataset as _AwA2Dataset
from goggles.data.cub.dataset import CUBDataset as _CUBDataset


DATASET_MAP = {
    'awa2': _AwA2Dataset,
    'cub': _CUBDataset
}

DATA_DIR_MAP = {
    'awa2': AWA2_DATA_DIR,
    'cub': CUB_DATA_DIR
}
