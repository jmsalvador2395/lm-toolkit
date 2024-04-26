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

class BERT(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 n_layers,
                 n_vocab,
                 seq_len=512,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm_eps=1e-05,
                 norm_first=False,
                 bias=True,
                 device=None,
                 dtype=None,
                 **kwargs):

        super(BERT, self).__init__()

        dtype = tensor_utils.get_dtype(dtype)

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

        self.token_emb = nn.Embedding(
            n_vocab,
            d_model,
        )
        self.position_emb = nn.Parameter(
            torch.rand(seq_len, d_model)
        )
        self.segment_emb = nn.Embedding(
            2,
            d_model,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            n_layers,
        )
        self.mlm_cls = nn.Linear(dim_feedforward, n_vocab)
        self.nsp_cls = nn.Linear(dim_feedforward, 2)

    def forward(self, input_ids, attention_mask, segment_ids=None):
        """
        returns scores for MLM and NSP
        """

        N, S = attention_mask.shape
        pad_mask = ~attention_mask.to(torch.bool)

        # apply embeddings
        scores = self.token_emb(input_ids)
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        scores += self.segment_emb(segment_ids)
        scores += self.position_emb[None, :S]

        # pass through transformer
        scores = self.transformer(
            scores, 
            src_key_padding_mask=pad_mask,
        )

        mlm_scores = self.mlm_cls(scores)
        nsp_scores = self.nsp_cls(scores[:, 0, :])

        return mlm_scores, nsp_scores
