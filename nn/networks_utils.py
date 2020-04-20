"""
# copy pasted from my earlier MADE implementation
# https://github.com/karpathy/pytorch-made

Implements a Masked Autoregressive MLP, where carefully constructed
binary masks over weights ensure the autoregressive property.
"""

import torch
import torch.nn.functional as F
from torch import nn


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(self, features, context=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features + int(context), features),
            nn.BatchNorm1d(features, eps=1e-3),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.3),
            nn.Linear(features, features),
            nn.BatchNorm1d(features, eps=1e-3),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x, context=None):
        if context is not None:
            return self.net(torch.cat([x, context], dim=1)) + x
        return self.net(x) + x
