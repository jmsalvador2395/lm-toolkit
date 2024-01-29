"""
this module is for defining actor-critic networks
"""
# external imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

class ActorCriticMLP(nn.Module):
    
    def __init__(self, cfg):
        super(ActorCriticMLP, self).__init__()

        ############## assign variables from cfg ############## 
        
        in_size = cfg.model['in_size']
        out_size = cfg.model['out_size']
        hidden_size = cfg.model['hidden_size']
        dropout = cfg.model['dropout']
        num_mlp_layers = cfg.model['mlp_layers']

        #######################################################

        # define policy model
        mlp_layers = [
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm((hidden_size,)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]

        mlp_layers += [
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm((hidden_size,)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]*(num_mlp_layers - 1)

        mlp_layers.append(
            nn.Linear(
                hidden_size,
                out_size,
            )
        )
        self.actor = nn.Sequential(*mlp_layers)

        # define value network
        mlp_layers.pop(-1)
        mlp_layers.append(
            nn.Linear(
                hidden_size,
                1
            )
        )
        self.critic = nn.Sequential(*mlp_layers)

    def forward(self, state):

        # compute value
        value = self.critic(state)

        # compute probabilities and create distribution
        dist = Categorical(F.softmax(
            self.actor(state),
            dim=-1
        ))

        return dist, value




