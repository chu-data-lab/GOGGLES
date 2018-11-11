import torch
import torch.nn as nn
import torch.nn.functional as F


def get_squared_l2_distances_from_references(reference_embeddings, candidate_embeddings):
    assert len(reference_embeddings.size()) == len(candidate_embeddings.size()) == 2

    num_references = reference_embeddings.size(0)
    num_candidates = candidate_embeddings.size(0)
    embedding_dim = reference_embeddings.size(1)

    assert candidate_embeddings.size(1) == embedding_dim

    reference_embeddings_ = reference_embeddings \
        .unsqueeze(1) \
        .expand(num_references, num_candidates, embedding_dim)
    candidate_embeddings_ = candidate_embeddings \
        .unsqueeze(0) \
        .expand(num_references, num_candidates, embedding_dim)

    dists = torch.sum((reference_embeddings_ - candidate_embeddings_) ** 2, dim=2)
    return dists
