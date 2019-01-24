import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRATCH_DIR = os.path.join(BASE_DIR, '_scratch')
ALL_RUNS_DIR = os.path.join(SCRATCH_DIR, 'runs')
CUB_DATA_DIR = os.path.join(SCRATCH_DIR, 'CUB_200_2011')
CACHE_DIR = os.path.join(SCRATCH_DIR, 'cache')
ANIMALS_DIR = '/media/seagate/rtorrent/AwA2/Animals_with_Attributes2'

LOGS_DIR_NAME = 'logs'
IMAGES_DIR_NAME = 'images'
PROTOTYPES_DIR_NAME = 'prototypes'

MODEL_FILE_NAME = 'model.pt'
