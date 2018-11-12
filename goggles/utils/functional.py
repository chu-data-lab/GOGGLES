import torch
import torch.nn as nn
import torch.nn.functional as F


def get_squared_l2_distances_from_references(embeddings1, embeddings2):
    assert len(embeddings1.size()) == len(embeddings2.size()) == 2

    num_embeddings1 = embeddings1.size(0)
    num_embeddings2 = embeddings2.size(0)
    embedding_dim = embeddings1.size(1)

    assert embeddings2.size(1) == embedding_dim

    embeddings1_ = embeddings1 \
        .unsqueeze(1) \
        .expand(num_embeddings1, num_embeddings2, embedding_dim)
    embeddings2_ = embeddings2 \
        .unsqueeze(0) \
        .expand(num_embeddings1, num_embeddings2, embedding_dim)

    dists = torch.sum((embeddings1_ - embeddings2_) ** 2, dim=2)
    return dists  # num_embeddings1 x num_embeddings2
