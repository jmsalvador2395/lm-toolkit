# external imports
import torch
import numpy as np
from torch import nn
from torch.nn import functional as f
from typing import List

class TransformerEncDec(nn.Module):
    
    def __init__(self, cfg):
        super(TransformerEncDec, self).__init__()

        # set defaults if they don't exist
        self.dev = cfg.get('device', 'cpu')
        V = cfg['vocab_size']
        embedding_dim = cfg.get('embedding_dim', 512)
        num_encoder_layers = cfg.get('num_encoder_layers', 6)
        num_decoder_layers = cfg.get('num_decoder_layers', 6)
        nhead = cfg.get('nhead', 8)
        dim_feed_forward = cfg.get('dim_feed_forward', 512)

        if num_encoder_layers % 2 != 0 \
        or num_decoder_layers % 2 != 0:
            raise ValueError('transformer_layers should be divisible by 2')

        # embedding layer
        self.emb_enc = nn.Embedding(
            V,
            embedding_dim
        )

        self.emb_dec = nn.Embedding(
            V,
            embedding_dim
        )
        
        self.enc_dec = nn.Transformer(
            embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dim,
                dim_feed_forward,
            ),
            nn.ReLU(),
            nn.Linear(
                dim_feed_forward,
                V,
            )
        )

    def forward(self, input_ids, tgt_ids, input_attn_mask, tgt_attn_mask):
        src_emb = self.emb_enc(input_ids)
        tgt_emb = self.emb_dec(tgt_ids)

        scores = self.enc_dec(src_emb, tgt_emb, input_attn_mask, tgt_attn_mask)
        scores = self.classifier(scores)

        return scores
