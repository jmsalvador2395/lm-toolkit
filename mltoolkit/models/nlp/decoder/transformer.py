
# external imports
import torch
import numpy as np
from torch import nn
from torch.nn import functional as f
from typing import List
from torch.distributions import Categorical

# local imports
from mltoolkit.nn_modules import PositionalEncoding

class TransformerDecoder(nn.Module):
    
    def __init__(self, cfg):
        super(TransformerDecoder, self).__init__()

        cfg = cfg.model

        # set defaults if they don't exist
        self.dev = cfg.get('device', ['cpu'])
        embedding_dim = cfg.get('embedding_dim', 768)
        num_decoder_layers = cfg.get('num_decoder_layers', 6)
        num_hidden_layers = cfg.get('num_hidden_layers', 1)
        nhead = cfg.get('nhead', 8)
        dim_feed_forward = cfg.get('dim_feed_forward', 512)
        seq_len = cfg.get('seq_len', 512)
        mlp_dropout = cfg.get('mlp_dropout', 0.1)
        decoder_dropout = cfg.get('decoder_dropout', 0.1)
        out_size = cfg.get('out_size', 10)

        if num_decoder_layers % len(self.dev) != 0:
            raise ValueError('transformer_layers should be divisible by number of devices in list')
        cutoff = num_decoder_layers // len(self.dev)
        
        transformer_devs = []
        for step in range(num_decoder_layers):
            transformer_devs.append(
                self.dev[step//cutoff]
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

        self.classifier.append(nn.Linear(
            dim_feed_forward,
            out_size
        ))

        self.mlp = nn.Sequential(*self.classifier)

    def forward(self, embeddings, mask):

        # for masking see https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask

        N, S, E = embeddings.shape

        # apply positional encodings
        scores = self.pos(embeddings)

        scores = self.decoder(
            scores,
            scores,
            tgt_key_padding_mask=mask,
            memory_key_padding_mask=mask,
        )

        scores = self.mlp(scores)

        dist = Categorical(logits=scores)

        return dist
