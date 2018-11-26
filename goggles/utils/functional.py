from types import SimpleNamespace

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def pairwise_squared_euclidean_distances(embeddings1, embeddings2):
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


def pairwise_cosine_similarities(embeddings1, embeddings2):
    assert len(embeddings1.size()) == len(embeddings2.size()) == 2
    embedding_dim = embeddings1.size(1)
    assert embeddings2.size(1) == embedding_dim

    embeddings1_ = F.normalize(embeddings1, dim=1)
    embeddings2_ = F.normalize(embeddings2, dim=1)

    sims = torch.mm(embeddings1_, embeddings2_.t())
    return sims  # num_embeddings1 x num_embeddings2


def get_performance_metrics(true_labels, predicted_labels):
    a = accuracy_score(y_true=true_labels, y_pred=predicted_labels)
    p = precision_score(y_true=true_labels, y_pred=predicted_labels)
    r = recall_score(y_true=true_labels, y_pred=predicted_labels)
    f1 = f1_score(y_true=true_labels, y_pred=predicted_labels)

    performance_metrics = SimpleNamespace(
        accuracy_score=a,
        precision_score = p,
        recall_score = r,
        f1_score = f1)

    return performance_metrics