"""
this is an example trainer class that inherits TrainerBase
"""
# external imports
import torch
import numpy as np
import datasets
from tqdm import tqdm
from torch import nn
from datasets import Dataset
from torch.utils.data import DataLoader

# for typing
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Tuple, List, Dict, TypeVar
T = TypeVar('T')

# local imports
from mltoolkit.trainers.base import TrainerBase
from mltoolkit.utils import (
    files,
    strings,
    display,
)

class TrainerExample(TrainerBase):
    
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug)

    def init_data(self) -> DataLoader:

        cfg = self.cfg

        train_loader = None
        val_loader = None
        test_loader = None

        return train_loader, val_loader, test_loader

    def init_model(self):
        
        cfg = self.cfg

        model = None

        return model

    def init_optimizer(self) -> Tuple[Optimizer, LRScheduler]:

        cfg = self.cfg

        optimizer = None
        scheduler = None
        
        return (
            optimizer,
            scheduler
        )

    def init_aux(self):
        pass

    def train_step(self, model: nn.Module, batch: T) -> Tuple[torch.Tensor, Dict]:

        loss = None
        metrics = {
            'scalar': {
                'example_scalar': None
            },
            'image': {
                'example_image': None
            },
            'histogram': {
                'example_histogram': None
            },
            'scalars': {
                'example_scalars': None
            }, 
        }

        return loss, metrics

    def eval_step(self, model: nn.Module, batch: T, mode: str):

        metrics = {
            'example_scalar': None,
            'example_image': None,
            'example_histogram': None,
            'example_scalars': None,
        }

        return metrics

    def on_eval_end(self, metrics: Dataset, mode: str):

        target_metric = None
        metrics = {
            'scalar': {
                'example_scalar': np.mean(metrics['example_scalar'])
            },
            'image': {
                'example_image': metrics['example_image'][-1]
            },
            'histogram': {
                'example_histogram': np.concatenate(metrics['example_histogram'])
            },
            'scalars': {
                'example_scalars': None
            }, 
        }

        return target_metric, metrics
