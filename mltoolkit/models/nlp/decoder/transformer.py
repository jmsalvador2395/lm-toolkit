
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
        mlp_layers = cfg.get('mlp_layers', 1)
        nhead = cfg.get('nhead', 8)
        dim_feed_forward = cfg.get('dim_feed_forward', 512)
        seq_len = cfg.get('seq_len', 512)
        mlp_dropout = cfg.get('mlp_dropout', 0.1)
        decoder_dropout = cfg.get('decoder_dropout', 0.1)
        out_size = cfg.get('out_size', 10)

        self.pos = PositionalEncoding(
            embedding_dim,
        )

        decoder_layer = nn.TransformerEncoderLayer(
                embedding_dim,
                nhead,
                dim_feed_forward,
                dropout=decoder_dropout,
                activation='relu',
                batch_first=True,
            )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_decoder_layers,
        )

        classifier= []
        for layer in range(mlp_layers):
            input_size = embedding_dim if layer == 0 else dim_feed_forward
            classifier.append(
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
            nn.init.kaiming_uniform_(classifier[-1][0].weight)

        classifier.append(nn.Linear(
            dim_feed_forward,
            out_size
        ))

        self.mlp = nn.Sequential(*classifier)

    def forward(self, embeddings, mask):

        # for masking see https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask

        N, S, E = embeddings.shape

        # apply positional encodings
        scores = self.pos(embeddings)

        """
        scores = self.decoder(
            scores,
            scores,
            tgt_key_padding_mask=mask,
            memory_key_padding_mask=mask,
        )
        """
        scores = self.decoder(
            scores,
            src_key_padding_mask=mask,
        )

        scores = self.mlp(scores)

        dist = Categorical(logits=scores)

        return dist
