"""
This code allows you to computing CKA (https://arxiv.org/abs/1905.00414) similarity using pytorch.
The code is adapted from https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment
"""

import math
import numpy as np
import torch

def centering(K):
    n = K.size(0)
    unit = torch.ones(n,n).cuda()
    I = torch.eye(n).cuda()
    H = I - unit/n
    return torch.mm(torch.mm(H, K), H)

def linear_HSIC(X, Y):
    L_X = torch.mm(X, X.transpose(0,1) )
    L_Y = torch.mm(Y, Y.transpose(0,1) )
    return torch.sum(centering(L_X) * centering(L_Y))

def rbf(X, sigma=None):
    GX = torch.mm(X, X.transpose(0,1) )
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).transpose(0,1) 
    if sigma is None:
        # mdist = torch.median(KX[KX != 0])
        try:
            mdist = torch.median(KX[KX != 0])
        except:
            mdist = torch.zeros(1).to(KX.device)
        sigma = math.sqrt(mdist.clamp(min=1e-12))
    KX = KX * (-0.5 / (sigma * sigma))
    KX = torch.exp(KX)
    return KX

def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X).clamp(min=1e-12))
    var2 = torch.sqrt(linear_HSIC(Y, Y).clamp(min=1e-12))

    return hsic / (var1 * var2)

def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma).clamp(min=1e-12))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma).clamp(min=1e-12))

    return hsic / (var1 * var2)

