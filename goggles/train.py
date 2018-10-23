from itertools import chain
import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from constants import *
from data.cub.dataset import CUBDataset
from models.encoder import Encoder
from models.decoder import Decoder


def main():
    dataset = CUBDataset(root=CUB_DATA_DIR,
                         species1_id=14, species2_id=90,
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.Scale((64, 64)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                         is_training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                             collate_fn=lambda b: torch.stack([d[0] for d in b]))

    netE = Encoder(64)
    netD = Decoder(netE.num_out_channels)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(chain(netE.parameters(), netD.parameters()))

    pbar = tqdm(range(100))
    for epoch in pbar:
        total_loss = 0.
        for i, x in enumerate(dataloader, 1):
            netE.zero_grad()
            netD.zero_grad()

            x = torch.autograd.Variable(x)
            z = netE(x)
            x_ = netD(z)

            loss = criterion(x_, x)
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()

        pbar.set_description('average loss in epoch %d was %0.04f' % (epoch + 1, total_loss / i))

        for i, x in enumerate(dataloader, 1):
            x = torch.autograd.Variable(x)
            z = netE(x)
            x_ = netD(z)

            reconstructed = x_.data.cpu().numpy()
            reconstructed = 255. * (0.5 + (reconstructed / 2.))
            reconstructed = np.array(reconstructed, dtype=np.uint8)

            for j, image in enumerate(reconstructed, 1):
                image = np.swapaxes(image, 0, 1)
                image = np.swapaxes(image, 1, 2)
                image = Image.fromarray(image)
                image.save(os.path.join('/Users/nilakshdas/Box/dev/GOGGLES/out', '%d-%d.png' % (i, j)))


if __name__ == '__main__':
    main()
