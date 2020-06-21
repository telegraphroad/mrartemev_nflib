"""
# copy pasted from my earlier MADE implementation
# https://github.com/karpathy/pytorch-made

Implements a Masked Autoregressive MLP, where carefully constructed
binary masks over weights ensure the autoregressive property.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class NoneLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, depth=1,
                 activation=nn.ReLU, batchnorm=nn.BatchNorm1d, context_dim=0):
        super().__init__()
        self.block = [
            nn.Linear(in_features + context_dim, out_features),
            activation(),
            batchnorm(out_features)
        ]
        for i in range(1, depth):
            self.block.append([
                nn.Linear(out_features, out_features),
                activation(),
                batchnorm(out_features)
            ])
        self.block = nn.Sequential(*self.block)

    def forward(self, x, context=None):
        if context is not None:
            return self.block(torch.cat([x, context], dim=1))
        return self.block(x)


class ARMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, depth=1,
                 activation=nn.ReLU, batchnorm=nn.BatchNorm1d, context_dim=0):
        super().__init__()
        self.block = [
            MaskedLinear(in_features, out_features),
            activation(),
            batchnorm(out_features)
        ]
        for i in range(1, depth):
            self.block.append([
                MaskedLinear(out_features, out_features),
                activation(),
                batchnorm(out_features)
            ])
        self.block = nn.Sequential(*self.block)

        if context_dim > 0:
            self.context_block = MLPBlock(
                context_dim, out_features, depth=depth,
                activation=activation, batchnorm=batchnorm, context_dim=0
            )

    def forward(self, x, context=None):
        if context is not None:
            return self.context_block(context) * self.block(x)
        return self.block(x)


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.zeros(out_features, in_features))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
