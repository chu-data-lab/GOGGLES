import torch.nn as nn


class Prototypes(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(Prototypes, self).__init__(*args, **kwargs)
