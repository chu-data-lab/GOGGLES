# GOGGLES

GOGGLES is a system for automatically generating probabilistic labels for image datasets based on the affinity coding paradigm. The paper can be found at https://arxiv.org/abs/1903.04552

## Installation and Setup

### Setup dependencies
```bash
pip install -r requirements.txt
```

### Download the data
```bash
bash tools/get_cub_dataset.sh _scratch
```

## Example Usage
See the default configuration:

```bash
python goggles/train.py print_config
```

Run a training experiment:
```bash
python goggles/train.py with \
  dataset=cub \
  filter_class_ids=14,90 \
  num_epochs=25000 \
  loss_lambda=0.01
```
