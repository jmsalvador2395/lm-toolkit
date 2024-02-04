# external imports
import torch
import numpy as np
import tokenizers
from torch import nn
from torch.nn import functional as f
from typing import List
from transformers import AutoTokenizer

# local imports
from mltoolkit.nn_modules import PositionalEncoding

class AutoLM(nn.Module):
    
    def __init__(self, cfg):
        super(AutoLM, self).__init__()

        # read params
        d_embed = cfg.params['d_embed']
        n_layers = cfg.params['n_transformer_layers']
        n_head = cfg.params['n_head']
        dropout = cfg.params['dropout']
        dim_feed_forward = cfg.params['dim_feed_forward']
        tokenizer_name = cfg.params['tokenizer']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.embeds = nn.Embedding(
            len(tokenizer),
            embedding_dim=d_embed,
        )

        xformer_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_head,
            dim_feedforward=dim_feed_forward,
            batch_first=True,
        )

        self.xformer = nn.TransformerEncoder(
            xformer_layer,
            num_layers=n_layers,
        )

    def forward(self, input_ids, attn_mask, pad_mask):

        emb = self.embeds(input_ids)

        return self.xformer(
            emb, 
            mask=attn_mask,
            src_key_padding_mask=pad_mask,
            is_causal=True,
        )

