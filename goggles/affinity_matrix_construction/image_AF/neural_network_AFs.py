import os
import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from goggles.utils.constants import *
from .pretrained_models.vgg import Vgg16

_make_cuda = lambda x: x.cuda() if torch.cuda.is_available() else x

class Context:
    def __init__(self, model, dataset, layer_idx):
        self.model = model
        self.dataset = dataset

        self._layer_idx = layer_idx
        self._model_out_dict = dict()

    def get_model_output(self, image_idx):
        if image_idx not in self._model_out_dict:
            x = self.dataset[image_idx]
            x = x.view((1,) + x.size())
            x = _make_cuda(torch.autograd.Variable(x, requires_grad=False))

            self._model_out_dict[image_idx] = \
                self.model.forward(x, layer_idx=self._layer_idx)[0]

        z = self._model_out_dict[image_idx]
        return z


def _get_patches(z, patch_idxs, normalize=False):
    """
    z: CxHxW
    patch_idxs: K
    """

    c = z.size(0)

    patches = z.view(c, -1).t()[patch_idxs]
    if normalize:
        patches = F.normalize(patches, dim=1)

    return patches


def _get_most_activated_channels(z, num_channels=5):
    """
    z: CxHxW
    """

    per_channel_max_activations, _ = z.max(1)[0].max(1)

    most_activated_channels = \
        torch.topk(per_channel_max_activations, num_channels)[1]

    return most_activated_channels


def _get_most_activated_patch_idxs_from_channels(z, channel_idxs):
    """
    z: CxHxW
    channel_idxs: K
    """

    k = channel_idxs.shape[0]

    most_activated_patch_idxs = \
        z[channel_idxs].view(k, -1).max(1)[1]

    l = list(most_activated_patch_idxs.cpu().numpy())
    d_ = {p: k - i - 1 for i, p in enumerate(reversed(l))}
    d = [(k - i - 1, p) for i, p in enumerate(reversed(l))]
    d = list(sorted(d))
    r,u = list(zip(*d))
    #u = sorted(d.keys(), key=lambda p:d[p])
    #r = [d[p] for p in u]
    return _make_cuda(torch.LongTensor(u)), \
        _make_cuda(torch.LongTensor(r))


def _get_score_matrix_for_image(image_idx, num_max_proposals, context):
    score_matrix = list()
    column_ids = list()

    z = context.get_model_output(image_idx)
    num_patches = z.size(1) * z.size(2)
    ch = _get_most_activated_channels(z, num_channels=num_max_proposals)
    pids, ranks = _get_most_activated_patch_idxs_from_channels(z, ch)
    proto_patches = _get_patches(z, pids, normalize=True)

    for patch_idx, rank in zip(pids.cpu().numpy(), ranks.cpu().numpy()):
        column_ids.append([image_idx, patch_idx, rank])

    for image_idx_ in range(len(context.dataset)):
        z_ = context.get_model_output(image_idx_)
        img_patches = _get_patches(z_, range(num_patches), normalize=True)
        scores = torch.matmul(img_patches, proto_patches.t()).max(0)[0]
        scores = scores.cpu().numpy()
        score_matrix.append(scores)
    return np.array(score_matrix), column_ids


def nn_AFs(dataset,layer_idx, num_max_proposals,cache=False):
    print('loading model...')
    model = _make_cuda(Vgg16())

    context = Context(
        model=model,
        dataset=dataset,
        layer_idx=layer_idx)
    if cache:
        out_filename = '.'.join([
            'v2',
            f'vgg16_layer{layer_idx:02d}',
            f'k{num_max_proposals:02d}',
            'scores.npz'])
        out_dirpath = os.path.join(SCRATCH_DIR, 'scores')
        os.makedirs(out_dirpath, exist_ok=True)
        out_filepath = os.path.join(out_dirpath, out_filename)

        print('saving output to %s' % out_filepath)
    affinity_matrix_list = [[] for _ in range(num_max_proposals)]
    all_column_ids = list()
    for image_idx in trange(len(context.dataset)):
        scores, cols = _get_score_matrix_for_image(
            image_idx, num_max_proposals, context)
        for i in range(min(num_max_proposals,scores.shape[1])):
            affinity_matrix_list[i].append(scores[:,i])
        #all_scores_matrix = np.concatenate(
        #    (all_scores_matrix, scores), axis=1)
        all_column_ids += cols
        #np.savez(
        #    out_filepath, version=2,
        #    scores=all_scores_matrix, cols=all_column_ids,
        #    num_max_proposals=num_max_proposals)
    for i in range(num_max_proposals):
        affinity_matrix_list[i] = np.array(affinity_matrix_list[i]).T
    return affinity_matrix_list