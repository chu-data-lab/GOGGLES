import torch.nn as nn


class Patch(nn.Module):
    def __init__(self, corner_idx, patch_size):
        super(Patch, self).__init__()

        self.corner_idx = corner_idx  # NW corner index
        self.patch_size = patch_size

        self.i_nw, self.j_nw = corner_idx

    def forward(self, z):
        i_nw, j_nw = self.corner_idx
        patch_w, patch_h = self.patch_size
        i_se, j_se = (i_nw + patch_h - 1,
                      j_nw + patch_w - 1)

        q = z[:, :, i_nw:i_se + 1, j_nw:j_se + 1]
        q = q.contiguous().view(q.size(0), -1)
        return q

    @staticmethod
    def from_spec(input_size, patch_size):
        input_w, input_h = input_size[-2:]
        patch_w, patch_h = patch_size

        patches = list()
        for i in range(input_h - patch_h + 1):
            for j in range(input_w - patch_w + 1):
                patches.append(Patch((i, j), patch_size))

        return patches


if __name__ == '__main__':
    import torch
    z = torch.IntTensor([[[[ 1,  2,  3,  4],  # 1 of 2
                           [ 5,  6,  7,  8],
                           [ 9, 10, 11, 12],
                           [13, 14, 15, 16]]],

                         [[[21, 22, 23, 24],  # 2 of 2
                           [25, 26, 27, 28],
                           [29, 30, 31, 32],
                           [33, 34, 35, 36]]]])

    print(z.size())

    patches = Patch.from_spec(z.size(), (2, 2))
    for patch in patches:
        print(patch(z).numpy())
        print('---')
