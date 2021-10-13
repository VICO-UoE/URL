"""
This code allows you to use adaptors for aligning features 
between multi-domain learning network and single domain learning networks.
The code is adapted from https://github.com/VICO-UoE/KD4MTL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class adaptor(torch.nn.Module):
    def __init__(self, num_datasets, dim_in, dim_out=None, opt='linear'):
        super(adaptor, self).__init__()
        if dim_out is None:
            dim_out = dim_in
        self.num_datasets = num_datasets

        for i in range(num_datasets):
            if opt == 'linear':
                setattr(self, 'conv{}'.format(i), torch.nn.Conv2d(dim_in, dim_out, 1, bias=False))
            else:
                setattr(self, 'conv{}'.format(i), nn.Sequential(
                    torch.nn.Conv2d(dim_in, 2*dim_in, 1, bias=False),
                    torch.nn.ReLU(True),
                    torch.nn.Conv2d(2*dim_in, dim_out, 1, bias=False),
                    )
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        results = []
        for i in range(self.num_datasets):
            ad_layer = getattr(self, 'conv{}'.format(i))
            if len(list(inputs[i].size())) < 4:
                input_ = inputs[i].view(inputs[i].size(0), -1, 1, 1)
            else:
                input_ = inputs[i]
            results.append(ad_layer(input_).flatten(1))
            # results.append(ad_layer(inputs[i]))
        return results







