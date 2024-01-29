import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

class SofsatExtractor(nn.Module):
    
    def __init__(self, cfg):
        super(SofsatExtractor, self).__init__()

        ##################### assign defaults ##################### 
        in_size = cfg.model.get('in_size', 2304)
        lstm_in_size = cfg.model.get('lstm_in_size', 1024)
        hidden_size = cfg.model.get('hidden_size', 768)
        num_layers = cfg.model.get('num_layers', 1)
        dropout = cfg.model.get('dropout', 0.1)
        out_size = cfg.model.get('out_size', 2)
        num_mlp_layers = cfg.model.get('mlp_layers', 2)
        ########################################################### 

        ##################### build model ##################### 

        # projection layer to get shape for lstm
        self.proj = nn.Linear(in_size, lstm_in_size)

        # define lstm
        self.lstm = nn.LSTM(
                lstm_in_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
        )

        # define classifier layer
        mlp_layers = [
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm((hidden_size,)),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
        ]*num_mlp_layers

        mlp_layers.append(
            nn.Linear(
                hidden_size,
                out_size,
            )
        )

        self.mlp = nn.Sequential(*mlp_layers)

        ####################################################### 

    def forward(self, batch):
        scores = self.proj(batch)
        scores, _ = self.lstm(scores)
        scores = self.mlp(scores)
        scores = F.log_softmax(scores, dim=-1)

        return scores

