'''
pa.py
Created by Wei-Hong Li [https://weihonglee.github.io]
This code allows you to attach pre-classifier alignment (PA) mapping to a pretrained backbone
and learn it on the support set to adapt features to a discriminative space.
'Universal Representation Learning from Multiple Domains for Few-shot Classification'
(https://arxiv.org/pdf/2103.13841.pdf)
'''
import torch
import numpy as np
import math
import torch.nn.init as init

from models.model_utils import sigmoid, cosine_sim
from models.losses import prototype_loss
from utils import device
import torch.nn.functional as F

def apply_selection(features, vartheta):
    """
    Performs pre-classifier alignment of features (feature adaptation) via a linear transformation.
    """

    features = features.unsqueeze(-1).unsqueeze(-1)
    features = F.conv2d(features, vartheta[0]).flatten(1)

    return features


def pa(context_features, context_labels, max_iter=40, ad_opt='linear', lr=0.1, distance='cos'):
    """
    PA method: learning a linear transformation per task to adapt the features to a discriminative space 
    on the support set during meta-testing
    """
    input_dim = context_features.size(1)
    output_dim = input_dim
    stdv = 1. / math.sqrt(input_dim)
    vartheta = []
    if ad_opt == 'linear':
        vartheta.append(torch.eye(output_dim, input_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True))

    optimizer = torch.optim.Adadelta(vartheta, lr=lr) 
    for i in range(max_iter):
        optimizer.zero_grad()
        selected_features = apply_selection(context_features, vartheta)
        loss, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features, context_labels, distance=distance)

        loss.backward()
        optimizer.step()
    return vartheta
