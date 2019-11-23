import torch
import os
import numpy as np
from ..transforms.transforms import DeNormalize


from torchvision import transforms

__all__ = ['get_scrambler']


def get_scrambler(config, device='cpu'):
    return Scrambler(config, device)


class Scrambler():
    def __init__(self, cfg, device):
        super(Scrambler, self).__init__()
        self.device = torch.device(device)

    def rotate_single(self, x):
        t = torch.zeros(x.size(), dtype=torch.float32)

        rotation = torch.randint(4, ())

        if rotation == 1:
            t = torch.transpose(x, 1, 2)
        elif rotation == 2:
            t = torch.flip(x, [1, 2])
        elif rotation == 3:
            t = torch.flip(torch.transpose(x, 1, 2), [1, 2])
        else:
            t[:, :, :] = x

        return t, rotation
