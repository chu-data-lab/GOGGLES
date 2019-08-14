import os as _os


BASE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

SCRATCH_DIR = _os.path.join(BASE_DIR, '_scratch')
ALL_RUNS_DIR = _os.path.join(SCRATCH_DIR, 'runs')
CACHE_DIR = _os.path.join(SCRATCH_DIR, 'cache')
RETRAINED_MODELS_DIR = _os.path.join(SCRATCH_DIR, 'models-retrained')

AWA2_DATA_DIR = _os.path.join(SCRATCH_DIR, 'AwA2')
CHNCXR_DATA_DIR = _os.path.join(SCRATCH_DIR, 'ShenzhenXRay')
CUB_DATA_DIR = _os.path.join(SCRATCH_DIR, 'CUB_200_2011')
GTSRB_DATA_DIR = _os.path.join(SCRATCH_DIR, 'gtsrb')
KAGGLE_CHEST_XRAY_DIR = _os.path.join(SCRATCH_DIR, 'kaggle', 'chest_xray')
SURFACE_DATA_DIR = _os.path.join(SCRATCH_DIR, 'surface_dataset')

LOGS_DIR_NAME = 'logs'
IMAGES_DIR_NAME = 'images'
PROTOTYPES_DIR_NAME = 'prototypes'

MODEL_FILE_NAME = 'model.pt'
