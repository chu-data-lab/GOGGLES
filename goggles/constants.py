import os as _os

from goggles.data.awa2.dataset import AwA2Dataset as _AwA2Dataset
from goggles.data.cub.dataset import CUBDataset as _CUBDataset


BASE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

SCRATCH_DIR = _os.path.join(BASE_DIR, '_scratch')
ALL_RUNS_DIR = _os.path.join(SCRATCH_DIR, 'runs')
CUB_DATA_DIR = _os.path.join(SCRATCH_DIR, 'CUB_200_2011')
AWA2_DATA_DIR = _os.path.join(SCRATCH_DIR, 'AwA2')
CACHE_DIR = _os.path.join(SCRATCH_DIR, 'cache')

LOGS_DIR_NAME = 'logs'
IMAGES_DIR_NAME = 'images'
PROTOTYPES_DIR_NAME = 'prototypes'

MODEL_FILE_NAME = 'model.pt'

DATASET_MAP = {
    'awa2': _AwA2Dataset,
    'cub': _CUBDataset
}

DATA_DIR_MAP = {
    'awa2': AWA2_DATA_DIR,
    'cub': CUB_DATA_DIR
}
