"""
this is an example trainer class that inherits TrainerBase
"""
# external imports
import torch
import numpy as np
import datasets
from torch import nn
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# for typing
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Tuple, List, Dict, TypeVar
T = TypeVar('T')

# local imports
from mltoolkit.templates import Trainer
from mltoolkit.utils import (
    files,
    strings,
    display,
)
from .model import TransformerAE
from .data_module import get_dataloaders

class TrainerTransformerAE(Trainer):
    def __init__(self, config_path, debug=False, accelerator=None):
        super().__init__(
            config_path,
            debug,
            accelerator=accelerator
        )

    def setup(self):
        cfg = self.cfg

        train_loader, val_loader = get_dataloaders(cfg)

        self.tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/all-mpnet-base-v2'
        )

        # define model
        model = TransformerAE(n_vocab = len(self.tokenizer), **cfg.params)

        # optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.params['lr'],
            weight_decay=self.cfg.params['weight_decay']
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.params['sched_step_size'],
            gamma=cfg.params['sched_gamma'],
        )

        self.loss_fn = nn.CrossEntropyLoss()

        return {
            'ae': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def step(self, batch: T, mode='train'):

        tokens = self.tokenizer(
            batch['text'],
            truncation=True,
            max_length=self.cfg.params['seq_len'],
            return_token_type_ids=False,
            padding=True,
            return_tensors='pt',
        ).to(self.accel.device)

        input_ids = tokens['input_ids'].clone()
        attention_mask = tokens['attention_mask']

        mask_prob = self.cfg.params['mask_prob']
        token_mask = torch.rand(input_ids.shape ,device=input_ids.device) < mask_prob
        #num_masked = torch.sum(mask_prob) # might use this later
        tokens['input_ids'][token_mask] = self.tokenizer.mask_token_id

        # compute scores and calculate loss
        scores = self.train_vars['ae'](**tokens)
        scores = scores['logits']

        # compute loss
        input_ids = input_ids.to(scores.device)
        attention_mask = attention_mask.to(scores.device)

        scores = scores[attention_mask == 1]
        labels = input_ids[attention_mask == 1]
        loss = self.loss_fn(
            scores,
            labels,
        )

        acc = torch.mean((torch.argmax(scores, dim=-1) == labels).to(torch.float32))

        return loss, acc 

    def train_step(self, batch: T) -> Tuple[torch.Tensor, Dict]:
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

        loss, acc = self.step(batch, mode='train')

        return loss, {
            'scalar' : {
                'loss' : loss,
                'accuracy': acc,
            }
        }

    def eval_step(self, batch: T, loader_name):
        """
        same as train_step but batch comes from previously assigned val_loader (test_loader evaluation not implemented yet)
        NOTE: in this function, torch.Tensor values need to be converted to float/int 

        Input
            batch[T]: a batch sampled from the previously assigned train dataloader
            mode[str]: string that will be either 'val' or 'train' depending on the procedure. 

        Return
            metrics[Dict]: the metrics you would like to track. Each run of eval_step() will 
                accumulate these metrics into a Dataset object and will be used as input 
                to on_eval_end() for final aggregation
        """
        #loss, accuracy = self.step(batch, mode='val')

        loss, acc = self.step(batch, mode='val')

        return {
            'loss': float(loss),
            'accuracy': float(acc),
        }

    def on_eval_end(self, metrics: List, mode: str):
        """
        use this to aggregate your metrics collected from the outputs of eval_step()

        Input:
            metrics[Dict[str, list]]: the set of metrics collected from eval_step()

        Return
            target_metric[T]: the metric that will be used to determine if the current model should be 
                checkpointed to {cfg.params['results']}/best_model.pt.
            log_values[Dict[Dict[str, T]]]: 
        """

        metrics_ds = Dataset.from_list(metrics['val_loader'])
        loss = np.mean(metrics_ds['loss'])
        acc = np.mean(metrics_ds['accuracy'])
        #accuracy = np.mean(metrics_ds['accuracy'])

        return acc, {
            'scalar' : {
                'loss' : loss,
                'accuracy': acc,
            }
        }
