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
    tensor_utils,
)
from .model import BERT
from . import data_module

class TrainerBERT(Trainer):
    def __init__(self, config_path, debug=False, accelerator=None):
        super().__init__(
            config_path,
            debug,
            accelerator=accelerator
        )

    def setup(self):
        cfg = self.cfg

        self.ds, train_loader, val_loader = data_module.get_dataloaders(cfg)

        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        model = BERT(**cfg.params, n_vocab=len(self.tokenizer))
        display.note(f'trainable params: {tensor_utils.count_trainable_params(model):,}')

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
            'bert': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def step(self, batch: T, mode='train'):

        # set some vars
        mode = 'train' if mode == 'train' else 'validation'
        bs = len(batch['text'])
        dslen = len(self.ds[mode])

        # set up nsp
        ns_ids = batch['row_id']+1
        ns_mask = torch.rand((bs,), device=ns_ids.device) > self.cfg.params['next_sent_prob']
        ns_ids[ns_mask] = torch.randint(
            0,
            dslen,
            (torch.sum(ns_mask),),
            device=ns_ids.device
        )
        next_sents = self.ds[mode][ns_ids]['text']
        ns_labels = ~ns_mask

        # pair sentences
        paired_sents = [s1 + '[SEP]' + s2 for s1, s2 in zip(batch['text'], next_sents)]

        tokens = self.tokenizer(paired_sents,
                                truncation=True,
                                max_length=self.cfg.params['seq_len'],
                                return_token_type_ids=False,
                                padding=True,
                                return_tensors='pt').to(self.accel.device)

        # filter out examples that don't fit both sentences
        valid_samples = torch.sum(tokens['input_ids'] == self.tokenizer.sep_token_id, dim=-1) == 2
        if not torch.all(valid_samples):
            tokens['input_ids'] = tokens['input_ids'][valid_samples]
            tokens['attention_mask'] = tokens['attention_mask'][valid_samples]
            ns_labels = ns_labels[valid_samples]

        # make segment ids
        labels = tokens['input_ids']
        rows, cols = torch.where(tokens['input_ids'] == self.tokenizer.sep_token_id)
        rows = rows[range(0, len(rows), 2)]
        cols = cols[range(0, len(cols), 2)]

        segment_ids = torch.zeros_like(tokens['input_ids'])
        segment_ids[rows, cols+1] = 1
        segment_ids = torch.cumsum(segment_ids, dim=-1)


        # TODO mask tokens
        mlm_labels = tokens['input_ids'].clone()
        mlm_targets = np.random.rand(*tokens['input_ids'].shape) < self.cfg.params['mlm_prob']
        mlm_targets *= tokens['attention_mask'].to(torch.bool).cpu().numpy()

        # do masking and replacement
        target_types = np.random.choice(
            [1, 2, 3],
            size=tokens['input_ids'].shape,
            p=self.cfg.params['mask_replace_keep_dist'],
        )
        target_types *= mlm_targets
        num_replace = np.sum(target_types == 2)

        # to masking
        tokens['input_ids'][target_types == 1] = self.tokenizer.mask_token_id

        # do token replacement
        tokens['input_ids'][target_types == 2] = torch.randint(
            0, 
            len(self.tokenizer), 
            (num_replace,),
            device=tokens['input_ids'].device,
        )

        mlm_scores, nsp_scores = self.train_vars['bert'](**tokens, segment_ids=segment_ids)

        # compute_loss
        nsp_loss = self.loss_fn(nsp_scores, ns_labels.to(torch.int64))

        if not np.any(mlm_targets):
            breakpoint()
        mlm_targets = torch.tensor(mlm_targets, device=mlm_scores.device)
        mlm_loss = self.loss_fn(mlm_scores[mlm_targets], mlm_labels[mlm_targets])

        loss = mlm_loss + nsp_loss

        return (
            loss,
            float(mlm_loss),
            float(nsp_loss),
        )

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

        loss, mlm_loss, nsp_loss = self.step(batch, mode='train')

        return loss, {
            'scalar' : {
                'loss' : loss,
                'mlm_loss': mlm_loss,
                'nsp_loss': nsp_loss,
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

        loss, mlm_loss, nsp_loss = self.step(batch, mode='val')

        return {
            'loss': float(loss),
            'mlm_loss': float(mlm_loss),
            'nsp_loss': float(nsp_loss),
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
        mlm_loss = np.mean(metrics_ds['mlm_loss'])
        nsp_loss = np.mean(metrics_ds['nsp_loss'])

        return loss, {
            'scalar' : {
                'loss': loss,
                'mlm_loss': mlm_loss,
                'nsp_loss': nsp_loss,
            }
        }
