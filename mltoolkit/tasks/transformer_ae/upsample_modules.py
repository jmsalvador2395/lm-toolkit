# external imports
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from typing import List

# local imports
from mltoolkit.utils import (
    display,
    tensor_utils
)

class UpConv(nn.Module):
    
    def __init__(self, 
                 d_input: int,
                 deconv_seq: List[int]=[16, 64, 128],
                 deconv_kernels: List[int]=[5, 9, 33],
                 deconv_activation: str='relu',
                 **kwargs):
        
        assert len(deconv_seq) == len(deconv_kernels), 'length of deconv_seq should match deconv_kernels'

        super(UpConv, self).__init__()

        # initialize up-conv layers
        deconv_seq = [1] + deconv_seq
        modules = []
        for in_channels, out_channels, kernel in zip(deconv_seq[:-1], 
                                                     deconv_seq[1:],
                                                     deconv_kernels):
            modules.append(nn.Conv1d(
                in_channels,
                out_channels,
                kernel,
                stride=1,
                padding=(kernel-1)//2,
            ))
            if deconv_activation == 'relu':
                modules.append(nn.ReLU())
            elif deconv_actionation == 'gelu':
                modules.append(nn.GELU())

        modules.append(nn.Linear(
            d_input,
            d_input, 
        ))

        self.deconv = nn.Sequential(*modules)

    def forward(self, X):
        return self.deconv(X[:, None, :])

class UpLinear(nn.Module):
    def __init__(self, 
                 d_input: int,
                 uplin_seq_len: int=256,
                 uplin_activation: str='relu',
                 **kwargs):
        super(UpLinear, self).__init__()

        modules = []
        modules.append(nn.Linear(
            d_input,
            d_input*uplin_seq_len
        ))
        modules.append(nn.Unflatten(-1, (uplin_seq_len, d_input)))

        if uplin_activation != 'none':
            if uplin_activation == 'relu':
                modules.append(nn.ReLU())
            elif uplin_activation == 'gelu':
                modules.append(nn.GELU())
            else:
                raise ValueError(f'invalid uplin_activation "{uplin_activation}"')
            modules.append(nn.Linear(d_input, d_input))
        self.model = nn.Sequential(*modules)


    def forward(self, X):
        return self.model(X)
