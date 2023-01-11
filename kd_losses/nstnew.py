from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import numpy as np
from pylab import *
# '''
# NSTNEW with Polynomial Kernel, where d=2 and c=0
# It can be treated as matching the Gram matrix of two vectorized feature map.
# '''

'''
NSTNEW with Polynomial Kernel, where d=2 and c=0
'''


class NSTNEW(nn.Module):
    '''
	Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
	https://arxiv.org/pdf/1707.01219.pdf
	'''

    def __init__(self):
        super(NSTNEW, self).__init__()
    
    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_s = F.normalize(fm_s, dim=2)

        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        fm_t = F.normalize(fm_t, dim=2)
        lossSkew = self.computeSkew(fm_s, fm_t)
        losscov=self.covariance(fm_s, fm_t)
        Originalloss=0.5*(self.poly_kernel(fm_t, fm_t).mean() + self.poly_kernel(fm_s, fm_s).mean() - 2 * self.poly_kernel(fm_s,fm_t).mean())
        Modefifyloss =  Originalloss + 0.25 * losscov + lossSkew * 0.2
        return Modefifyloss

    def covariance(self,fm_s, fm_t):
        fm_s = fm_s.view(1, -1)
        fm_s = fm_s.detach().cpu().numpy()
        fm_t = fm_t.view(1, -1)
        fm_t = fm_t.detach().cpu().numpy()
        covloss=np.cov(fm_s[0],fm_t[0])
        return np.absolute(covloss[0][1])
   


    def computeSkew(self, fm_s, fm_t):
        fm_s = fm_s.view(1, -1)
        fm_s = fm_s.detach().cpu().numpy()
        fm_t = fm_t.view(1, -1)
        fm_t = fm_t.detach().cpu().numpy()
        s = pd.Series(fm_s[0])
        t = pd.Series(fm_t[0])
        loss = np.absolute(s.skew() - t.skew())
        return loss

    def poly_kernel(self, fm1, fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1).pow(2)

        return out