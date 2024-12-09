"""
positional encoding class taken from:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
import torch
import math
from torch import nn, Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, 
        d_model: int, 
        dropout: float=0.1, 
        seq_len: int=5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LearnedEncoding(nn.Module):
    def __init__(self, 
        d_model: int, 
        dropout: float=0.1, 
        seq_len: int=512
    ):
        super().__init__()
        self.positions = nn.Parameter(torch.randn(seq_len, d_model))
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        """
        N, L, D = x.shape
        x += self.positions[None, :L]
        return self.dropout(x)
