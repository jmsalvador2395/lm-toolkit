
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
        self.dev = cfg.get('device', ['cpu'])
        V = cfg['vocab_size']
        embedding_dim = cfg.get('embedding_dim', 512)
        num_decoder_layers = cfg.get('num_decoder_layers', 6)
        num_hidden_layers = cfg.get('num_hidden_layers', 1)
        nhead = cfg.get('nhead', 8)
        dim_feed_forward = cfg.get('dim_feed_forward', 512)
        seq_len = cfg.get('seq_len', 512)
        mlp_dropout = cfg.get('mlp_dropout', 0.1)
        decoder_dropout = cfg.get('decoder_dropout', 0.1)

        if num_decoder_layers % len(self.dev) != 0:
            raise ValueError('transformer_layers should be divisible by number of devices in list')
        cutoff = num_decoder_layers // len(self.dev)
        
        transformer_devs = []
        for step in range(num_decoder_layers):
            transformer_devs.append(
                self.dev[step//cutoff]
            )

        # embedding layer
        self.emb = nn.Embedding(
            V,
            embedding_dim,
        )

        self.pos = PositionalEncoding(
            embedding_dim,
        )

        decoder_layer = nn.TransformerDecoderLayer(
                embedding_dim,
                nhead,
                dim_feed_forward,
                dropout=decoder_dropout,
                activation='gelu',
                batch_first=True,
            )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
        )

        self.classifier= []
        for layer in range(num_hidden_layers):
            input_size = embedding_dim if layer == 0 else dim_feed_forward
            self.classifier.append(
                nn.Sequential(
                    nn.Linear(
                        input_size,
                        dim_feed_forward,
                    ),
                    #nn.LayerNorm(
                        #(dim_feed_forward,),
                    #),
                    nn.ReLU(),
                    nn.Dropout(p=mlp_dropout),
                )
            )
            nn.init.kaiming_uniform_(self.classifier[-1][0].weight)
        self.classifier = nn.ModuleList(self.classifier)

        self.proj = nn.Linear(
            dim_feed_forward,
            V,
        )

    def forward(self, input_ids, attn_mask):
        N, S = input_ids.shape
        # get embeddings
        out = self.emb(input_ids)

        # apply positional encodings
        out = self.pos(out)

        attn_mask = \
            torch.triu(torch.ones(S, S, dtype=torch.bool)).to(out.device)
        attn_mask.fill_diagonal_(False)

        """
        # feed to transformer
        for step, dec in enumerate(self.decoder):
            out = dec(
                out,
                out,
                tgt_is_causal=True,
                memory_is_causal=True,
            )
        """
        out = self.decoder(
            out,
            out,
            attn_mask,
            attn_mask,
        )

        # get classification scores
        for cls in self.classifier:
            out = cls(out)

        out = self.proj(out)

        return out
