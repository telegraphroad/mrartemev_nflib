"""
Predefined Networks
"""

import torch
from torch import nn
from .networks_utils import ResidualBlock, get_mask, MaskedLinear


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, in_features, out_features, hidden_features=32, depth=4, context=False):
        """ context  - int, False or zero if None"""
        super().__init__()
        self.net = []
        self.net.append(nn.Linear(in_features + int(context), hidden_features))
        self.net.append(nn.LeakyReLU(0.01))
        for _ in range(depth):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.LeakyReLU(0.01))
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, context=None):
        if context is not None:
            return self.net(torch.cat([context, x], dim=1))
        return self.net(x)


class ARMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=32, depth=4, context=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        """

        super().__init__()
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        input_mask = get_mask(in_features, hidden_features, in_features, mask_type='input')
        hidden_mask = get_mask(hidden_features, hidden_features, in_features)
        output_mask = get_mask(hidden_features, out_features, in_features,
                               mask_type='output')

        self.net = []
        self.net.append(MaskedLinear(in_features, hidden_features, input_mask))
        self.net.append(nn.LeakyReLU(0.01))
        for _ in range(depth):
            self.net.append(MaskedLinear(hidden_features, hidden_features, hidden_mask))
            self.net.append(nn.LeakyReLU(0.01))
        self.net.append(MaskedLinear(hidden_features, out_features, output_mask))
        self.net = nn.Sequential(*self.net)

        if context:
            self.context_net = MLP(int(context), out_features, hidden_features, depth=depth)

    def forward(self, x, context=None):
        if context is not None:
            return torch.sigmoid(self.context_net(context)) * self.net(x)
        return self.net(x)


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self, in_features, out_features, hidden_features=32, depth=4, context=False):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(in_features + int(context), hidden_features),
            nn.BatchNorm1d(hidden_features, eps=1e-3),
            nn.LeakyReLU(0.01)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(
                features=hidden_features,
                context=context,
            ) for _ in range(depth//2)
        ])
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=1)
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x, context=context)
        outputs = self.final_layer(x)
        return outputs
