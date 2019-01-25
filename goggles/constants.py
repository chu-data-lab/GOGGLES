import os as _os


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
