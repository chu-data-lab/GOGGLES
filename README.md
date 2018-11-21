# GOGGLES

## Installation and Setup

### Setup Dependencies
```bash
pip install -r requirements.txt
```

## Example Usage
See the default configuration:

```bash
python goggles/train.py print_config
```

Run a training experiment:
```bash
python goggles/train.py with \
  filter_class_ids=14,90 \
  num_epochs=25000 \
  loss_lambda=0.01
```
