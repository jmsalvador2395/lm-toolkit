import torch
from torch import nn
import random
import numpy as np

from torch import Tensor
from typing import List, Dict, Tuple, Optional, Iterable, Callable


def get_dl_params(seed):
    """
    returns a seedworker and generator for initialize torch dataloaders.
    this is used to control the randomness of the trainers

    Input:
    - seed[int]: the RNG seed
    
    Output:
    - Callable: the seed worker function to pass to the dataloader __init__ function
    - torch.Generator: the random number generator to pass to the dataloader __init__ function
    """

    # define function for RNG in dataloaders
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)

    return seed_worker, g

def get_causal_mask(S: int, device='cpu') -> torch.Tensor:
    """
    returns a 2d causal attention mask of shape (S, S)

    Input:
    - S[int]: the sequence length of the mask you want to make

    Output:
    - torch.Tensor: causal attention mask of shape (S, S)
    """

    attn_mask = torch.triu(
        torch.ones(
            (S, S), 
            device=device,
            dtype=torch.bool
        ), 
        diagonal=1,
    )

    return attn_mask

def count_params(model: nn.Module) -> int:
    """
    returns the count of parameters in a torch nn.Module

    Input:
    - model[nn.Module]: the model we want the parameter count of

    Output:
    - int: the count of parameters in the model
    """
    return sum(p.numel() for p in model.parameters())

def count_trainable_params(model: nn.Module) -> int:
    """
    returns the count of learnable parameters in a torch nn.Module

    Input:
    - model[nn.Module]: the model we want the parameter count of

    Output:
    - int: the count of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pad_and_vstack(tensors:     Iterable[Tensor], 
                   pad_dim:     int=0,
                   pad_value:   int|float=0) -> Tensor:
    """
    pads a set of tensors along pad_dim using pad_value and then vertical stacks them

    Input:
    - tensors[Iterable[Tensor]]: a collection of tensors
    - pad_dim[int]: the dimension that you want to apply padding along (default: 0)
    - pad_value: the value to pad with (default: 0)

    Output:
    - Tensor: a single torch tensor
    """
    breakpoint()
