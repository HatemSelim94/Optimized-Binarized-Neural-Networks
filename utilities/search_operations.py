import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class Bilinear(nn.Upsample):
    def __init__(self, size= None, scale_factor = 2, mode: str = 'bilinear', align_corners = False):
        super(Bilinear, self).__init__(size, scale_factor, mode, align_corners)


class Cat(nn.Module):
    def __init__(self, dim=1, not_bin=0):
        super(Cat, self).__init__()
        self.dim = dim
        self.not_bin = not_bin

    def forward(self, x):
        return torch.cat(x, self.dim)


class Sum(nn.Module):
    def __init__(self, dim=1, not_bin=0):
        super(Sum, self).__init__()
        self.dim = dim
        self.not_bin = not_bin
    def forward(self, x):
        return sum(x)
