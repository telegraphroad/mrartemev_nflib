import logging

import numpy as np
import torch
from torch import nn

from scipy.stats import gennorm
import pandas as pd
import numpy as np
import torch
from torch.distributions import ExponentialFamily,Categorical,constraints
from torch.distributions.utils import _standard_normal,broadcast_all
from numbers import Real, Number
import math
import copy
from scipy.stats import gennorm
import numpy as np
import pandas as pd
import torch
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions import Categorical
from torch.distributions import constraints
from typing import Dict

import warnings
import os
import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from scipy.stats import gennorm

import warnings

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
import copy
from math import prod
import logging
from numpy import save
import torch.distributions as tdistr
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.distributions import MultivariateNormal


from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


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

class GenNormal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.
    Example::
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])
    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'p': constraints.real}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def exponent(self):
        return self.p

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale,p, validate_args=None):
        self.loc, self.scale, self.p = broadcast_all(loc, scale, p)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(GenNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GenNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.p = self.p.expand(batch_shape)
        super(GenNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    def rsample(self, sample_shape=torch.Size()):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        shape = self._extended_shape(sample_shape)
        #print('sample shape',sample_shape)
        #print('shape',shape)
        ipower = 1.0 / self.p
        ipower = ipower.cpu()
        gamma_dist = torch.distributions.Gamma(ipower, 1.0)
        
        gamma_sample = gamma_dist.rsample(shape).cpu()
        #print('gs shape',gamma_sample.shape)
        binary_sample = torch.randint(low=0, high=2, size=shape, dtype=self.loc.dtype) * 2 - 1
        #print('bs shape',binary_sample.shape)
        
        if len(binary_sample.shape) ==  len(gamma_sample.shape) - 1:
            gamma_sample = gamma_sample.squeeze(len(gamma_sample.shape) - 1)
            #print('bingo!')
        #print('bs shape',binary_sample.shape)
        sampled = binary_sample * torch.pow(torch.abs(gamma_sample), ipower)
        #print('sampled shape',sampled.shape)
        print(self.loc.item(),':::::',self.scale.item(),':::::',self.p.item())
        return self.loc.to(device) + self.scale.to(device) * sampled.to(device)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        #print('shape',shape)
        with torch.no_grad():
            ipower = 1.0 / self.p
            ipower = ipower#.cpu()
            gamma_dist = torch.distributions.Gamma(ipower, 1.0)
            gamma_sample = gamma_dist.sample(shape)#.cpu()
            binary_sample = (torch.randint(low=0, high=2, size=shape, dtype=self.loc.dtype) * 2 - 1)
            if (len(gamma_sample.shape) == len(binary_sample.shape) + 1) and gamma_sample.shape[-1]==gamma_sample.shape[-2]:
              gamma_sample = gamma_dist.sample(shape[0:-1])#.cpu()
              #print('=================================================================================================================')
            #print(binary_sample)
            #print(gamma_sample)
            sampled = binary_sample.squeeze() * torch.pow(torch.abs(gamma_sample.squeeze()), torch.FloatTensor(ipower))
            #print(self.loc.item(),':::::',self.scale.item(),':::::',self.p.item())
            return self.loc + self.scale * sampled

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        return (-((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi)))


    
class NormalizingFlowModelGGD(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self,rep_sample, flows,loc,scale,p):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        #self.prior = prior
        self.flows = nn.ModuleList(flows)
        self._dim = None
        self._rep_sample = rep_sample
        self.loc = nn.Parameter(torch.zeros(())+loc)
        self.scale = nn.Parameter(torch.zeros(())+scale)
        self.p = nn.Parameter(torch.zeros(())+p)
        self.loc.requires_grad = True
        self.scale.requires_grad = True
        self.p.requires_grad = True
        self.prior = GenNormal(loc=self.loc,scale=self.scale,p=self.p)
        
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

class NormalizingFlowModelMVN(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self,rep_sample, flows,loc,cov,dim):
        super().__init__()
        self.register_buffer('placeholder', torch.randn(1))
        #self.prior = prior
        self.flows = nn.ModuleList(flows)
        self._dim = None
        self._rep_sample = rep_sample
        self.loc = nn.Parameter(torch.zeros(())+loc)
        self.cov = nn.Parameter(torch.eye((dim))+cov)
        
        self.loc.requires_grad = True
        self.cov.requires_grad = True
        
        
        self.prior = MultivariateNormal(self.loc, self.cov)
        
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

