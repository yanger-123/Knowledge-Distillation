from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# '''
# NST with Polynomial Kernel, where d=2 and c=0
# It can be treated as matching the Gram matrix of two vectorized feature map.
# '''


'''
NST with Polynomial Kernel, where d=2 and c=0
'''
class STandNST(nn.Module):
    def __init__(self, T):
        super(STandNST, self).__init__()
        self.T = T


    def forward(self, fm_s, fm_t,flag):
        if flag==0:
            return NSTa.forward(fm_s,fm_t)
        else:
            return SoftTarget.forward(self,fm_s, fm_t)

class NSTa(nn.Module):
    '''
	Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
	https://arxiv.org/pdf/1707.01219.pdf
	'''

    def __init__(self):
        super(NSTa, self).__init__()
 
    def forward(fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_s = F.normalize(fm_s, dim=2)

        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        fm_t = F.normalize(fm_t, dim=2)
        lossSkew = NSTa.computeSkew(fm_s, fm_t)
        losscov = NSTa.covariance(fm_s, fm_t)
        Originalloss = 0.5 * (NSTa.poly_kernel(fm_t, fm_t).mean() + NSTa.poly_kernel(fm_s, fm_s).mean() - 2 * NSTa.poly_kernel(fm_s, fm_t).mean())
        Modifyloss = Originalloss + 0.25 * losscov  + lossSkew * 0.2
        return Modifyloss

    def covariance(fm_s, fm_t):
        fm_s = fm_s.view(1, -1)
        fm_s = fm_s.detach().cpu().numpy()
        fm_t = fm_t.view(1, -1)
        fm_t = fm_t.detach().cpu().numpy()
        covloss = np.cov(fm_s[0], fm_t[0])
        return np.absolute(covloss[0][1])

    def computeSkew(fm_s, fm_t):
        fm_s = fm_s.view(1, -1)
        fm_s = fm_s.detach().cpu().numpy()
        fm_t = fm_t.view(1, -1)
        fm_t = fm_t.detach().cpu().numpy()
        s = pd.Series(fm_s[0])
        t = pd.Series(fm_t[0])
        loss = np.absolute(s.skew() - t.skew())
        return loss

    def poly_kernel(fm1, fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1).pow(2)

        return out

    def linear_kernel(fm1,fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1)
        return out

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):

		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss