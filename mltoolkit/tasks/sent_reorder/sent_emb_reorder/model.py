# external imports
import torch
import numpy as np
import transformers
import math
from torch import nn
from torch.nn import functional as f
from typing import List
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

# local imports
from mltoolkit.nn_modules import PositionalEncoding
from mltoolkit.utils import (
    display,
    tensor_utils
)

class SentEmbedReorder(nn.Module):
    
    def __init__(self, 
                 n_vocab,
                 n_decoder_layers=6,
                 dropout=0.1,
                 freeze_encoder=True,
                 pool_op='average',
                 d_pool=None,
                 upsampling_proc='linear',
                 **kwargs):

        super(SentEmbedReorder, self).__init__()

    def forward(self, X):
        pass
