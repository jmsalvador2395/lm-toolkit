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

class TrainerMNIST(TrainerBase):
    
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug)

    def init_data(self) -> DataLoader:
        """
        This function is used to intialize your dataloaders for the training loop

        Return:
            train_loader[Dataloader]: the dataloader used in the training loop. batches will be fed to train_step() function
            val_loader[Dataloader]: the dataloader used in the validation loop. batches will be fed to eval_step() function
            test_loader[Dataloader]: the dataloader used for final testing. NOT YET IMPLEMENTED
        """

        cfg = self.cfg

        # load mnist dataset
        self.ds = datasets.load_dataset(
            'mnist',
            cache_dir=cfg.data['cache_dir']
        ).with_format('torch')
        in_size = cfg.model['hidden_dim']

        # preprocess
        def normalize(batch):
            batch.update({
                'image' : batch['image'].to(torch.float32)/255.
            })
            return batch

        ds = self.ds.map(
            normalize,
            batched=True,
            batch_size=1000,
            num_proc=cfg.data['num_proc'],
        )

        train_loader = DataLoader(
            ds['train'],
            shuffle=cfg.data['shuffle'],
            batch_size=cfg.data['batch_size'],
        )

        val_loader = DataLoader(
            ds['test'],
            shuffle=cfg.data['shuffle'],
            batch_size=cfg.data['batch_size'],
        )

        return train_loader, val_loader, None

    def init_model(self):
        """
        use this function to initialize your model.
        feel free to initialize any other models here and just assign them as self.<other model>

        Return
            model[nn.Module]: the model used for training
        """
        
        cfg = self.cfg

        # define model
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                cfg.model['in_size'], 
                cfg.model['hidden_dim']
            ),
            torch.nn.BatchNorm1d(cfg.model['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.model['dropout']),
            torch.nn.Linear(
                cfg.model['hidden_dim'],
                cfg.model['num_classes'],
            )
        ).to(self.dev)

        return model

    def init_optimizer(self) -> Tuple[Optimizer, LRScheduler]:
        """
        initialize your optimizer and learning rate schedulers here

        Return
            optim_tools[Tuple[Optimizer, LRScheduler]: a tuple that includes the optimizer and scheduler
        """

        cfg = self.cfg

        # optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.optim['lr'],
            weight_decay=self.cfg.optim['weight_decay']
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.optim['sched_step_size'],
            gamma=cfg.optim['sched_gamma'],
        )
        
        return (
            optimizer,
            scheduler
        )

    def init_aux(self):
        """
        Use this function to initialize any other important variables that you want to use for training.
        """
        self.loss_fn = nn.CrossEntropyLoss()

    def step(self, model, batch, mode='train'):

        # compute scores and calculate loss
        scores = model(batch['image'].to(self.dev))
        labels = batch['label'].to(self.dev)
        loss = self.loss_fn(scores, labels)
        accuracy = torch.sum(torch.argmax(scores, dim=-1) == labels)/len(labels)

        return loss, accuracy

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

        loss, accuracy = self.step(model, batch, mode='train')

        return loss, {
            'scalar' : {
                'loss' : loss,
                'accuracy' : accuracy
            }
        }

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
        loss, accuracy = self.step(model, batch, mode=mode)

        return {
            'loss': loss,
            'accuracy': accuracy,
        }

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

        loss = np.mean(metrics['loss'])
        accuracy = np.mean(metrics['accuracy'])

        return accuracy, {
            'scalar' : {
                'loss' : loss,
                'accuracy' : accuracy
            }
        }
