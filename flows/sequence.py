import logging

import numpy as np
import torch
from torch import nn

logger = logging.getLogger('main.nflib.flows.SequenceFlows')


class InvertiblePermutation(nn.Module):
    # @robdrynkin
    def __init__(self, dim):
        super().__init__()
        self.perm = nn.Parameter(torch.randperm(dim), requires_grad=False)
        self.inv_perm = nn.Parameter(torch.argsort(self.perm), requires_grad=False)

    def forward(self, x, context=None):
        return x[:, self.perm], 0

    def inverse(self, z, context=None):
        return z[:, self.inv_perm], 0


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior,rep_sample, flows):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        self.prior = prior
        self.flows = nn.ModuleList(flows)
        self._dim = None
        self._rep_sample = rep_sample
        self.dist_p1 = None
        self.dist_p2 = None
        self.dist_p3 = None

    def forward(self, x, context=None):
        m, self._dim = x.shape
        log_det = torch.zeros(m, device=self.placeholder.device)
        for flow in self.flows:
            x, ld = flow.forward(x, context=context)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z, context=None):
        if len(z.shape)>2:
            z = z.squeeze()
        m, _ = z.shape
        log_det = torch.zeros(m, device=self.placeholder.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, context=context)
            log_det += ld
        x = z
        return x, log_det

    def log_prob(self, x):
        _, prior_logprob, log_det = self.forward(x)
        return prior_logprob + log_det

    def sample(self, num_samples, context=None):
        if type(self.prior) == torch.distributions.multivariate_normal.MultivariateNormal:
          if self._rep_sample:
            z = self.prior.rsample((num_samples,)).to(self.placeholder.device)
            
          else:
            z = self.prior.sample((num_samples,)).to(self.placeholder.device)
            
          #print('mvn')
        else:
          if self._rep_sample:
            z = self.prior.rsample((num_samples,self._dim)).to(self.placeholder.device)
            
          else:
            z = self.prior.sample((num_samples,self._dim)).to(self.placeholder.device)
            
          #print('ggd')
        x, _ = self.inverse(z, context=context)
        return x

