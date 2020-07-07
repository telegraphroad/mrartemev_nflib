"""
Predefined Networks
"""
import numpy as np
import torch
from torch import nn
from .networks_utils import MLPBlock, ARMLPBlock, NoneLayer, MaskedLinear


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=32, depth=5, context_dim=0):
        super().__init__()
        self.net = [nn.Linear(in_features + int(context_dim), hidden_features),
                    nn.BatchNorm1d(hidden_features),
                    nn.ReLU()]
        for _ in range(1, depth - 1):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.BatchNorm1d(hidden_features))
            self.net.append(nn.ReLU())
        self.net[-1] = nn.Tanh()
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, context=None):
        if context is not None:
            return self.net(torch.cat([context, x], dim=1))
        return self.net(x)


class ARMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=32, depth=5, context_dim=0):
        super().__init__()
        assert out_features % in_features == 0, "nout must be integer multiple of nin"
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.depth = depth

        self.net = [MaskedLinear(in_features, hidden_features),
                    nn.BatchNorm1d(hidden_features),
                    nn.ReLU()]
        for i in range(1, depth - 1):
            self.net.append(MaskedLinear(hidden_features, hidden_features))
            self.net.append(nn.BatchNorm1d(hidden_features))
            self.net.append(nn.ReLU())
        self.net[-1] = nn.Tanh()
        self.net.append(MaskedLinear(hidden_features, out_features))

        self.net = nn.Sequential(*self.net)
        if context_dim > 0:
            self.context_net = MLP(context_dim, out_features, hidden_features, depth)
        self.seed = 0  # for cycling through num_masks orderings
        self.update_masks()  # builds the initial m_dict connectivity

    def update_masks(self):

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)

        # sample the order of the inputs and the connectivity of all neurons
        m_dict = {-1: np.arange(self.in_features)}

        for l in range(self.depth):
            m_dict[l] = rng.randint(m_dict[l - 1].min(), self.in_features - 1, size=self.hidden_features)
        # construct the mask matrices
        masks = [m_dict[l - 1][:, None] <= m_dict[l][None, :] for l in range(self.depth-1)]
        masks.append(m_dict[self.depth - 1][:, None] < m_dict[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.out_features > self.in_features:
            k = self.out_features // self.in_features
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for ind, (l, m) in enumerate(zip(layers, masks)):
            l.set_mask(m)

    def forward(self, x, context=None):
        if context is not None:
            return torch.sigmoid(self.context_net(context)) * self.net(x)
        return self.net(x)
