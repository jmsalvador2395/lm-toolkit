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
        """
        initialization function. ideally there is no need to edit this since initialization is performed at designated functions

        Input:
            config_path[str]: the location of the YAML config. parent class takes care of loading the config.
            debug[bool]: used to set debug-related parameters. affected parameters will be printed to console
        """
        super().__init__(config_path, debug)

    def init_data(self) -> DataLoader:
        """
        This function is used to intialize your dataloaders for the training loop

        Return:
            train_loader[Dataloader]: the dataloader used in the training loop. batches will be fed to train_step() function
            val_loader[Dataloader]: the dataloader used in the validation loop. batches will be fed to eval_step() function
            test_loader[Dataloader]: the dataloader used for final testing. NOT YET IMPLEMENTED
        """

        # assign cfg to use freely
        cfg = self.cfg

        # initialize train dataloader
        train_loader = None

        # initialize validation dataloader
        val_loader = None

        # TODO build test set evaluation into TrainerBase
        # initialize test dataloader
        test_loader = None

        return train_loader, val_loader, test_loader

    def init_model(self):
        """
        use this function to initialize your model.
        feel free to initialize any other models here and just assign them as self.<other model>

        Return
            model[nn.Module]: the model used for training
        """
        
        # assign cfg to use freely
        cfg = self.cfg

        # define model
        model = None

        return model

    def init_optimizer(self) -> Tuple[Optimizer, LRScheduler]:
        """
        initialize your optimizer and learning rate schedulers here

        Return
            optim_tools[Tuple[Optimizer, LRScheduler]: a tuple that includes the optimizer and scheduler
        """

        # assign cfg to use freely
        cfg = self.cfg

        # optimizer
        optimizer = None

        # scheduler
        scheduler = None
        
        return (
            optimizer,
            scheduler
        )

    def init_aux(self):
        """
        Use this function to initialize any other important variables that you want to use for training.
        """
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, model: nn.Module, batch: T) -> Tuple[torch.Tensor, Dict]:
        """
        This function is used to compute the loss and training statistics

        Input
            model[nn.Module]: this is the model you previously assigned.
            batch[T]: a batch sampled from the previously assigned train dataloader

        Return
            loss[torch.Tensor]: the computed loss
            metrics[Dict]: the metrics you would like to track.
                refer to {project_root}/mltoolkit/trainers/base.py for the currently supported keyword trackers
        """

        # compute loss
        loss = None

        # compute other metrics
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
            'scalars': { # this is not recommended to uses. very messy
                'example_scalars': None
            }, 
        }

        return loss, metrics

    def eval_step(self, model: nn.Module, batch: T, mode: str):
        """
        same as train_step but batch comes from previously assigned val_loader (test_loader evaluation not implemented yet)

        Input
            model[nn.Module]: this will be either the regular model or the averaged model from SWA
            batch[T]: a batch sampled from the previously assigned train dataloader
            mode[str]: string that will be either 'val' or 'train' depending on the procedure. 

        Return
            metrics[Dict]: the metrics you would like to track. Each run of eval_step() will 
                accumulate these metrics into a Dataset object and will be used as input 
                to on_eval_end() for final aggregation
        """
        # compute loss
        loss = None

        metrics = {
            'example_scalar': None,
            'example_image': None,
            'example_histogram': None,
            'example_scalars': None,
        }

        return loss, metrics

    def on_eval_end(self, metrics: Dataset, mode: str):
        """
        use this to aggregate your metrics collected from the outputs of eval_step()

        Input:
            metrics[Dataset]: the set of metrics collected from eval_step()

        Return
            target_metric[T]: the metric that will be used to determine if the current model should be 
                checkpointed to {cfg.model['ckpt_dir']}/best_model.pt. Change cfg.model['keep_higher_eval']
                to True if you want to keep higher values and False to keep lower
        """

        # compute other metrics
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
            'scalars': { # this is not recommended to uses. very messy
                'example_scalars': None # figure it out lol
            }, 
        }
