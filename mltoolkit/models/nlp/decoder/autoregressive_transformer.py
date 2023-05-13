
# external imports
import torch
import numpy as np
from torch import nn
from torch.nn import functional as f
from typing import List

# local imports
from mltoolkit.nn_modules import PositionalEncoding

class AutoregressiveTransformerDecoder(nn.Module):
    
    def __init__(self, cfg):
        super(AutoregressiveTransformerDecoder, self).__init__()

        # set defaults if they don't exist
        self.dev = cfg.get('device', 'cpu')
        V = cfg['vocab_size']
        embedding_dim = cfg.get('embedding_dim', 512)
        num_decoder_layers = cfg.get('num_decoder_layers', 6)
        nhead = cfg.get('nhead', 8)
        dim_feed_forward = cfg.get('dim_feed_forward', 512)
        seq_len = cfg.get('seq_len', 512)

        if num_decoder_layers % 2 != 0:
            raise ValueError('transformer_layers should be divisible by 2')

        # embedding layer
        self.emb = nn.Embedding(
            V,
            embedding_dim
        )

        self.pos = PositionalEncoding(
            embedding_dim,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            embedding_dim,
            nhead,
            dim_feed_forward,
            batch_first=True,
        )

        self.decoder1 = nn.TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dim,
                dim_feed_forward,
            ),
            nn.LayerNorm(
                (seq_len, dim_feed_forward),
            ),
            nn.ReLU(),
            nn.Dropout(p=cfg.get('dropout', .1)),
            nn.Linear(
                dim_feed_forward,
                V,
            ),

        )

    def forward(self, input_ids, tgt_ids, input_attn_mask, tgt_attn_mask):

        # get embeddings
        emb = self.emb(input_ids)

        # apply positional encodings
        emb = self.pos(emb)

        # feed to transformer
        emb = self.decoder1(
            emb,
            emb,
            input_attn_mask,
            tgt_attn_mask
        )

        """
        emb = self.decoder2(
            emb,
            emb,
            input_attn_mask,
            tgt_attn_mask,
        )
        """

        # get classification scores
        scores = self.classifier(emb)

        return scores
