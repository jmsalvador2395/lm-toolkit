# external imports
import torch
import numpy as np
import transformers
import math
from torch import nn
from torch.nn import functional as f
from typing import List

# local imports
from mltoolkit.nn_modules import PositionalEncoding, LearnedEncoding
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
            num_xformer_layers=1,
            mlp_hidden_dim=2048,
            num_mlp_layers=2,
            with_positions=True,
            seq_len=128,
            **kwargs):

        super(SentEmbedReorder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_xformer_layers,
        )

        self.with_positions = with_positions
        if with_positions == 'learned':
            #self.positions = nn.Parameter(torch.randn(seq_len, d_model))
            self.positions = LearnedEncoding(
                d_model=d_model, dropout=dropout, seq_len=seq_len
            )
        elif with_positions == 'sinusoidal':
            self.positions = PositionalEncoding(
                d_model=d_model, dropout=dropout, seq_len=seq_len
            )
        elif with_positions is not None:
            raise ValueError(
                '`with_positions` should be either [`learned`, '
                '`sinusoidal`, None]'
            )

        act_fn = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
        }

        if num_mlp_layers > 0:
            module_list = [
                nn.Linear(d_model, mlp_hidden_dim), 
                nn.LayerNorm(mlp_hidden_dim),
                act_fn[activation](),
                nn.Dropout(dropout)
            ]
            module_list += (num_mlp_layers-1)*[
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim), 
                nn.LayerNorm(mlp_hidden_dim),
                act_fn[activation](),
                nn.Dropout(dropout)
            ]
            module_list.append(nn.Linear(mlp_hidden_dim, 1))

            self.output = nn.Sequential(*module_list)
        else:
            self.output = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, nonlinearity='relu'
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.Embedding, nn.Parameter)):
                nn.init.uniform_(layer.weight, -0.1, 0.1)
            elif isinstance(layer, nn.TransformerEncoderLayer):
                for sublayer in layer.children():
                    if isinstance(sublayer, nn.Linear):
                        nn.init.kaiming_uniform_(
                            sublayer.weight, nonlinearity='relu'
                        )
                        if sublayer.bias is not None:
                            nn.init.zeros_(sublayer.bias)
                    elif isinstance(sublayer, nn.LayerNorm):
                        nn.init.ones_(sublayer.weight)
                        nn.init.zeros_(sublayer.bias)

    def forward(self, X, mask):
        N, L, D = X.shape
        if self.with_positions:
            #X += self.positions[None, :L]
            X = self.positions(X)
        scores = self.encoder(X, src_key_padding_mask=mask)
        scores = self.output(scores)
        scores = scores.squeeze(-1)
        return scores


class SentEmbedReorderCls(nn.Module):
    
    def __init__(
            self, 
            n_class=5,
            d_model=1024,
            nhead=12,
            dim_feedforward=2048,
            activation='gelu',
            dropout=0.1,
            num_xformer_layers=1,
            mlp_hidden_dim=2048,
            num_mlp_layers=2,
            with_positions=True,
            seq_len=128,
            **kwargs):

        super(SentEmbedReorderCls, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_xformer_layers,
        )

        self.with_positions = with_positions
        if with_positions:
            self.positions = nn.Parameter(torch.randn(seq_len, d_model))

        module_list = [
            nn.Linear(d_model, mlp_hidden_dim), 
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        module_list += (num_mlp_layers-1)*[
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), 
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        module_list.append(nn.Linear(mlp_hidden_dim, n_class))

        self.output= nn.Sequential(*module_list)

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, nonlinearity='relu'
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.Embedding, nn.Parameter)):
                nn.init.uniform_(layer.weight, -0.1, 0.1)
            elif isinstance(layer, nn.TransformerEncoderLayer):
                for sublayer in layer.children():
                    if isinstance(sublayer, nn.Linear):
                        nn.init.kaiming_uniform_(
                            sublayer.weight, nonlinearity='relu'
                        )
                        if sublayer.bias is not None:
                            nn.init.zeros_(sublayer.bias)
                    elif isinstance(sublayer, nn.LayerNorm):
                        nn.init.ones_(sublayer.weight)
                        nn.init.zeros_(sublayer.bias)

    def forward(self, X, mask):
        N, L, D = X.shape
        if self.with_positions:
            X += self.positions[None, :L]
        scores = self.encoder(X, src_key_padding_mask=mask)
        scores = self.output(scores)
        return scores
