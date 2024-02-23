# external imports
import torch
import numpy as np
import transformers
import math
from torch import nn
from torch.nn import functional as f
from typing import List
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

# local imports
from mltoolkit.nn_modules import PositionalEncoding
from mltoolkit.utils import (
    display,
    tensor_utils
)
from .pool_modules import MeanPooling, AttentionPooling
from .upsample_modules import UpConv, UpLinear

class TransformerAE(nn.Module):
    
    def __init__(self, 
                 n_vocab,
                 n_decoder_layers=6,
                 dropout=0.1,
                 freeze_encoder=True,
                 pool_op='average',
                 d_pool=None,
                 upsampling_proc='linear',
                 **kwargs):

        super(TransformerAE, self).__init__()

        d_embed = 768
        self.S = kwargs['seq_len']

        # initialize encoder
        self.encoder = AutoModel.from_pretrained(
            'sentence-transformers/all-mpnet-base-v2'
        )
        if freeze_encoder:
            self.encoder = tensor_utils.freeze_module(self.encoder)

        # initialize pooler
        if pool_op == 'average':
            self.pooler = MeanPooling()
        elif pool_op == 'attention':
            self.pooler = AttentionPooling(d_input=d_embed,**kwargs)
        else:
            raise ValueError(f'invalid pooling operation "{pool_op}"')

        # initialize upsampler
        if upsampling_proc == 'conv':
            self.upsampler = UpConv(d_input=d_embed, **kwargs)
        elif upsampling_proc == 'linear':
            self.upsampler = UpLinear(d_input=d_embed, **kwargs)
        else:
            raise ValueError(
                f'invalid upsampling_proc (upsampling procedure) "{upsampling_proc}"'
            )

        # initaialize decoder (nn.Encoder used because i just want to do self-attention
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=kwargs['decoder_n_heads'],
            dim_feedforward=kwargs['decoder_d_ffn'],
            activation=kwargs['decoder_activation'],
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerEncoder(
            dec_layer,
            kwargs['decoder_n_layers'],
        )

        cls_activation = kwargs['cls_activation']
        cls_d_ffn = kwargs['cls_d_ffn']
        cls_modules = []
        if cls_activation  == 'relu':
            cls_modules.append(nn.Linear(d_embed, cls_d_ffn))
            cls_modules.append(nn.ReLU())
            cls_modules.append(nn.Linear(cls_d_ffn, n_vocab))
        elif cls_activation == 'gelu':
            cls_modules.append(nn.Linear(d_embed, cls_d_ffn))
            cls_modules.append(nn.GELU())
            cls_modules.append(nn.Linear(cls_d_ffn, n_vocab))
        elif cls_activation == 'none':
            cls_modules.append(nn.Linear(d_embed, n_vocab))
        else:
            raise ValueError(f'invalid cls_activation optino "{cls_activation}"')

        self.classifier = nn.Sequential(*cls_modules)



    def forward(self, input_ids, attention_mask):
        B, L =  attention_mask.shape

        scores = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sent_embeds = self.pooler(
            scores['last_hidden_state'],
            attention_mask,
        )

        upsamples = self.upsampler(sent_embeds)
        if upsamples.shape[-1] < self.S:
            N, s = upsamples.shape
            diff = self.S - s
            upsamples = torch.cat(
                (upsamples, torch.zeros(N, diff, device=upsamples.device)), 
                dim=-1
            )
        if attention_mask.shape[-1] < self.S:
            N, s = attention_mask.shape
            diff = self.S - s
            attention_mask = torch.cat(
                (attention_mask, torch.zeros(N, diff, device=attention_mask.device)), 
                dim=-1
            )

        last_hidden_state = self.decoder(
            upsamples, 
            src_key_padding_mask=(attention_mask == 0).to(upsamples.device),
        )
        last_hidden_state = last_hidden_state[:, :L]

        logits = self.classifier(last_hidden_state)

        return {
            'encodings': sent_embeds,
            'last_hidden_state': last_hidden_state,
            'logits': logits,
        }
