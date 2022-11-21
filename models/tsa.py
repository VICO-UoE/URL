'''
tsa.py
Created by Wei-Hong Li [https://weihonglee.github.io]
This code allows you to attach task-specific parameters, including adapters, pre-classifier alignment (PA) mapping
from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
(https://arxiv.org/pdf/2103.13841.pdf), to a pretrained backbone. 
It only learns attached task-specific parameters from scratch on the support set to adapt 
the pretrained model for previously unseen task with very few labeled samples.
'Cross-domain Few-shot Learning with Task-specific Adapters.' (https://arxiv.org/pdf/2107.00358.pdf)
'''

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
from config import args
import copy
import torch.nn.functional as F
from models.losses import prototype_loss
from utils import device

class conv_tsa(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters
        if 'alpha' not in args['test.tsa_opt']:
            self.ad_type = 'none'
        else:
            self.ad_type = args['test.tsa_ad_type']
            self.ad_form = args['test.tsa_ad_form']
        if self.ad_type == 'residual':
            if self.ad_form == 'matrix' or planes != in_planes:
                self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
            else:
                self.alpha = nn.Parameter(torch.ones(1, planes, 1, 1))
        elif self.ad_type == 'serial':
            if self.ad_form == 'matrix':
                self.alpha = nn.Parameter(torch.ones(planes, planes, 1, 1))
            else:
                self.alpha = nn.Parameter(torch.ones(1, planes, 1, 1))
            self.alpha_bias = nn.Parameter(torch.ones(1, planes, 1, 1))
            self.alpha_bias.requires_grad = True
        if self.ad_type != 'none':
            self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        if self.ad_type == 'residual':
            if self.alpha.size(0) > 1:
                # residual adaptation in matrix form
                y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
            else:
                # residual adaptation in channel-wise (vector)
                y = y + x * self.alpha
        elif self.ad_type == 'serial':
            if self.alpha.size(0) > 1:
                # serial adaptation in matrix form
                y = F.conv2d(y, self.alpha) + self.alpha_bias
            else:
                # serial adaptation in channel-wise (vector)
                y = y * self.alpha + self.alpha_bias
        return y

class pa(nn.Module):
    """ 
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        return x

class resnet_tsa(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet):
        super(resnet_tsa, self).__init__()
        self.tsa_init = args['test.tsa_init']
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta = pa(feat_dim)
        setattr(self, 'beta', beta)

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):

        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                # initialize each adapter as an identity matrix
                if self.tsa_init == 'eye':
                    if v.size(0) > 1:
                        v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
                    else:
                        v.data = torch.ones(v.size()).to(v.device)
                    # for residual adapter, each adapter is initialized as identity matrix scaled by 0.0001
                    if  args['test.tsa_ad_type'] == 'residual':
                        v.data = v.data * 0.0001
                    if 'bias' in k:
                        v.data = v.data * 0
                elif self.tsa_init == 'random':
                    # randomly initialization
                    v.data = torch.rand(v.data.size()).data.normal_(0, 0.001).to(v.device)
        # initialize pre-classifier alignment mapping (beta)
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)

def tsa(context_images, context_labels, model, max_iter=40, lr=0.1, lr_beta=1, distance='cos'):
    """
    Optimizing task-specific parameters attached to the ResNet backbone, 
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    model.eval()
    tsa_opt = args['test.tsa_opt']
    alpha_params = [v for k, v in model.named_parameters() if 'alpha' in k]
    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    params = []
    if 'alpha' in tsa_opt:
        params.append({'params': alpha_params})
    if 'beta' in tsa_opt:
        params.append({'params': beta_params, 'lr': lr_beta})

    optimizer = torch.optim.Adadelta(params, lr=lr) 

    if 'alpha' not in tsa_opt:
        with torch.no_grad():
            context_features = model.embed(context_images)
    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()

        if 'alpha' in tsa_opt:
            # adapt features by task-specific adapters
            context_features = model.embed(context_images)
        if 'beta' in tsa_opt:
            # adapt feature by PA (beta)
            aligned_features = model.beta(context_features)
        else:
            aligned_features = context_features
        loss, stat, _ = prototype_loss(aligned_features, context_labels,
                                       aligned_features, context_labels, distance=distance)

        loss.backward()
        optimizer.step()
    return