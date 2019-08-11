# GOGGLES

GOGGLES is a system for automatically generating probabilistic labels for image datasets based on the affinity coding paradigm. The paper can be found at https://arxiv.org/abs/1903.04552

## Installation

```bash
pip install git+https://github.com/chu-data-lab/GOGGLES.git@dev```
```

## Example Usage
```bash
from goggles import generate_labels
path_to_images = "data/images"
dev_set_indices = [1,10,11,25] #indices of images in the development set
dev_set_labels = [0,0,1,1] #the coresponding labels 
labels = generate_labels(path_to_images,dev_set_indices,dev_set_labels)
```

