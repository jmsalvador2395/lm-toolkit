import torch
from torch import nn
import random
import numpy as np

from torch import Tensor
from typing import List, Dict, Tuple, Optional, Iterable, Callable, Any

from . import display


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

def pad_and_stack(tensors:     Iterable[Tensor], 
                  stack_dim:   int=0,
                  pad_dim:     int=0,
                  pad_value:   torch.dtype=0) -> Tensor:
    """
    pads a set of tensors along pad_dim using pad_value and then vertical stacks them along stack_dim

    Input:
    - tensors[Iterable[Tensor]]: a collection of tensors
    - stack_dim[int]: the dimension to stack along
    - pad_dim[int]: the dimension that you want to apply padding along (default: 0)
    - pad_value: the value to pad with (default: 0)

    Output:
    - Tensor: a single torch tensor
    """

    # get maximum for padding dim
    dim_size = max([t.shape[pad_dim] for t in tensors])
    
    # pad and add to out_tensors list
    out_tensors = []
    for t in tensors:
        shape = list(t.shape)
        pad_dim_size = dim_size - shape[pad_dim]
        if pad_dim_size > 0:
            shape[pad_dim] = dim_size - shape[pad_dim]
            padding_tensor = torch.full(
                shape,
                pad_value,
                dtype=t.dtype,
                device=t.device,
            )
            out_tensors.append(torch.cat(
                [t, padding_tensor], 
                dim=pad_dim
            ))
        else:
            out_tensors.append(t)

    # stack
    out_tensors = torch.cat(out_tensors, dim=stack_dim)

    return out_tensors

def get_dtype(dtype_str: str) -> torch.dtype:
    """
    takes in a string and returns the corresponding torch dtype

    Input:
    - dtype_str[str]: the datatype specified as a string

    Output:
    - torch.dtype: the specified datatype
    """
    dtypes = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bool': torch.bool,
    }
    dtype = dtypes.get(dtype_str, None)

    if dtype is None:
        display.error(f'provided dtype ({dtype_str}) is invalid')
        raise ValueError()

    return dtype

def freeze_module(mod: nn.Module) -> nn.Module:
    """
    freezes the parameters of a  torch.nn.Module

    Input:
    - mod[nn.Module]: the torch module to freeze

    Output:
    - nn.Module: the frozen module
    """
    for param in mod.parameters():
        param.requires_grad=False

    return mod
