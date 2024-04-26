# external imports
import torch
import numpy as np
import transformers
import math
from torch import nn
from torch.nn import functional as f
from typing import List

# local imports
from mltoolkit.nn_modules import PositionalEncoding
from mltoolkit.utils import (
    display,
    tensor_utils
)

class SentEmbedReorder(nn.Module):
    
    def __init__(
            self, 
            d_model=1024,
            nhead=12,
            dim_feedforward=2048,
            activation='gelu',
            dropout=0.1,
            num_xformer_layers=3,
            mlp_hidden_dim=2048,
            num_mlp_layers=2,
            n_positions=None,
            **kwargs):

        super(SentEmbedReorder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_xformer_layers,
        )

        self.n_positions = n_positions
        if n_positions:
            self.positions = nn.Parameter(
                torch.randn(n_positions, d_model)
            )

        module_list = [nn.Linear(d_model, mlp_hidden_dim), nn.ReLU()]
        module_list += (num_mlp_layers-1)*[
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), 
            nn.ReLU()
        ]
        module_list.append(nn.Linear(mlp_hidden_dim, 1))

        self.mlp = nn.Sequential(*module_list)


    def forward(self, X, pad_mask):

        N, L, E = X.shape

        if self.n_positions:
            X += self.positions[:L][None]
        scores = self.encoder(X, src_key_padding_mask=pad_mask)
        scores = self.mlp(scores)
        scores = scores.squeeze(-1)
        return scores
