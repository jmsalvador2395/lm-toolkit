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
from .model import AutoLM

class TrainerAutoLM(Trainer):
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug)

    def setup(self):
        cfg = self.cfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.params['tokenizer']
        )

        # define model
        model = AutoLM(cfg)
        dtype = cfg.params['dtype']
        if dtype == 'float32':
            pass
        elif dtype == 'float16':
            model = model.half()
        elif dtype == 'bfloat16':
            model = model.to(torch.bfloat16)

        # load mnist dataset
        ds = datasets.load_dataset(
            'bookcorpus',
            cache_dir=cfg.paths['cache'],
            trust_remote_code=True,
        )
        ds = ds['train'].train_test_split(
            train_size=cfg.params['train_test_split']
        )

        train_loader = DataLoader(
            ds['train'],
            shuffle=cfg.params['shuffle'],
            batch_size=cfg.params['batch_size'],
        )

        val_loader = DataLoader(
            ds['test'],
            shuffle=cfg.params['shuffle'],
            batch_size=cfg.params['batch_size'],
        )

        # optimizer
        optimizer = torch.optim.Adam(
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
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def step(self, batch: T, mode='train'):

        tokens = self.tokenizer(
            batch['text'],
            truncation=True,
            max_length=self.cfg.params['seq_len']+1,
            return_token_type_ids=False,
            padding=True,
            return_tensors='pt',
        ).to(self.accel.device)

        input_ids = tokens['input_ids'][:, :-1]
        pad_ids = tokens['attention_mask'][:, :-1]

        pad_mask = (pad_ids == 0)
        _, L = pad_mask.shape
        attn_mask = torch.ones(
            (L, L),
            dtype=torch.bool
        )
        attn_mask = attn_mask.triu(diagonal=1)

        # compute scores and calculate loss
        scores = self.train_vars['model'](
            input_ids,
            attn_mask,
            pad_mask,
        )
        label_mask = tokens['attention_mask'][:, 1:] == 1
        labels = tokens['input_ids'][:, 1:][label_mask]
        scores = scores[label_mask]

        loss = self.loss_fn(scores, labels)
        perplexity = torch.exp(loss.detach())

        return loss, perplexity

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

        loss, perplexity = self.step(batch, mode='train')

        return loss, {
            'scalar' : {
                'loss' : loss,
                'perplexity': perplexity,
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

        loss, perplexity = self.step(batch, mode='val')

        return {
            'loss': float(loss),
            'perplexity': float(perplexity),
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
        perplexity = np.mean(metrics_ds['perplexity'])
        #accuracy = np.mean(metrics_ds['accuracy'])

        return perplexity, {
            'scalar' : {
                'loss' : loss,
                'perplexity': perplexity,
            }
        }
