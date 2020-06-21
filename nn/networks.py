"""
Predefined Networks
"""

import numpy as np
import torch
from torch import nn
from .networks_utils import MLPBlock, ARMLPBlock, NoneLayer, MaskedLinear


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, in_features, out_features, hidden_features=32, depth=4, context_dim=0):
        """ context  - int, None if None"""
        super().__init__()
        self.net = [MLPBlock(in_features, hidden_features, activation=nn.ELU, context_dim=context_dim)]
        for i in range(1, depth):
            if i < depth-2:
                self.net.append(MLPBlock(hidden_features, hidden_features,
                                         activation=nn.ELU, context_dim=context_dim))
            elif i == depth-2:
                self.net.append(MLPBlock(hidden_features, hidden_features,
                                         activation=nn.Tanh, context_dim=context_dim))
            elif i == depth-1:
                self.net.append(MLPBlock(hidden_features, out_features,
                                         activation=NoneLayer, context_dim=context_dim))
        self.net = nn.ModuleList(self.net)

    def forward(self, x, context=None):
        for layer in self.net:
            x = layer(x, context=context)
        return x


class ARMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=32, depth=4, context_dim=0):
        super().__init__()
        assert out_features % in_features == 0, "nout must be integer multiple of nin"
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.depth = depth

        self.net = [ARMLPBlock(in_features, hidden_features, activation=nn.ELU, context_dim=context_dim)]
        for i in range(1, depth):
            if i < depth-2:
                self.net.append(ARMLPBlock(hidden_features, hidden_features,
                                           activation=nn.ELU, context_dim=context_dim))
            elif i == depth-2:
                self.net.append(ARMLPBlock(hidden_features, hidden_features,
                                           activation=nn.Tanh, context_dim=context_dim))
            elif i == depth-1:
                self.net.append(ARMLPBlock(hidden_features, out_features,
                                           activation=NoneLayer, context_dim=context_dim))
        self.net = nn.ModuleList(self.net)
        self.seed = 0  # for cycling through num_masks orderings
        self.update_masks()  # builds the initial m_dict connectivity

    def update_masks(self):

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)

        # sample the order of the inputs and the connectivity of all neurons
        m_dict = dict()
        m_dict[-1] = np.arange(self.in_features)

        # set the masks in all MaskedLinear layers
        mask_num = 0
        last_sublayer = None
        for layer in self.net:
            for sublayer in layer.block:
                if isinstance(sublayer, MaskedLinear):
                    m_dict[mask_num] = rng.randint(m_dict[mask_num - 1].min(),
                                                   self.in_features - 1,
                                                   size=self.hidden_features)
                    sublayer_mask = m_dict[mask_num - 1][:, None] <= m_dict[mask_num][None, :]
                    mask_num += 1
                    print(sublayer.mask.shape, sublayer_mask.T.shape)
                    if sublayer.mask.shape == sublayer_mask.T.shape:
                        sublayer.set_mask(sublayer_mask)
                    last_sublayer = sublayer
        last_mask = m_dict[mask_num - 1][:, None] < m_dict[-1][None, :]
        if self.out_features > self.in_features:
            k = self.out_features // self.in_features
            # replicate the mask across the other outputs
            last_mask = np.concatenate([last_mask] * k, axis=1)
        last_sublayer.set_mask(last_mask)

    def forward(self, x, context=None):
        for layer in self.net:
            x = layer(x, context=context)
        return x
