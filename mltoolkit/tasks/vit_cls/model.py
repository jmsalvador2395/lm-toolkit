# external imports
import torch
import numpy as np
import transformers
from torch import nn
from torch.nn import functional as f
from typing import List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# local imports
from mltoolkit.nn_modules import PositionalEncoding
from mltoolkit.utils import (
    display,
    tensor_utils,
)

class Patchifier(nn.Module):
    """
    patchifies a set of input images
    Shape is assumed to be (N, W, H, C)

    N = batch_size
    W = width
    H = height
    C = # channels
    """

    def __init__(self,
                 patch_height,
                 patch_width,
                 **kwargs):

        super(Patchifier, self).__init__()

        self.Hp = patch_height
        self.Wp = patch_width

    def forward(self, images):

        N, W, H, C = images.shape
        Wp, Hp = self.Wp, self.Hp

        assert W%Wp == 0 and H%Hp == 0, 'patch dimensions do not evenly divide image'

        num_patches = (W*H*C)//(Wp*Hp*C)

        patches = images.unfold(1, Wp, Wp).unfold(2, Hp, Hp)
        patches = patches.permute(0, 1, 2, 4, 5, 3)
        patches = patches.reshape((N, num_patches, -1))

        return patches


class VitCls(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 n_layers,
                 n_cls,
                 patch_width,
                 patch_height,
                 seq_len,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm_eps=1e-05,
                 norm_first=False,
                 bias=True,
                 device=None,
                 dtype=None,
                 **kwargs):

        super(VitCls, self).__init__()

        dtype = tensor_utils.get_dtype(dtype)

        self.patchify = Patchifier(patch_width, patch_height)

        self.cls_emb = nn.Parameter(torch.randn(d_model, requires_grad=True))

        self.pos = nn.Parameter(torch.randn(
            seq_len+1, # +1 to account for cls embedding
            d_model,
            requires_grad=True
        ))

        layer = nn.TransformerEncoderLayer(
            d_model,
            n_head,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu', # BERT paper specifies gelu
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            n_layers,
        )
        self.classifier = nn.Linear(d_model, n_cls)

    def forward(self, X):
        """
        returns classification scores
        """

        # patchify
        patches = self.patchify(X)
        B, S, d = patches.shape

        # add cls embedding
        patches = torch.cat(
            [self.cls_emb.repeat(B, 1, 1), patches],
            dim=1
        )

        # apply position embeddings and classify
        patches += self.pos
        scores = self.transformer(patches)
        scores = self.classifier(scores)

        
        # extract cls embedding
        scores = scores[:, 0, :]

        return scores
