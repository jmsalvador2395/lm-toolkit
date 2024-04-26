"""
this is an example trainer class that inherits TrainerBase
"""
# external imports
import torch
from torch import nn
import numpy as np
from datasets import Dataset
from scipy.stats import kendalltau, spearmanr

# for typing
from typing import Tuple, List, Dict, TypeVar
T = TypeVar('T')

# local imports
from mltoolkit.templates import Trainer
from mltoolkit.utils import (
    files,
    strings,
    display,
)
from .model import SentEmbedReorder
from .data_module import get_dataloaders

class TrainerSentEmbedReordering5(Trainer):
    def __init__(self, config_path, debug=False, accelerator=None):
        super().__init__(
            config_path,
            debug,
            accelerator=accelerator
        )

    def setup(self):
        cfg = self.cfg

        train_loader, val_loader = get_dataloaders(cfg)

        # define model
        model = SentEmbedReorder(**cfg.params)

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

        self.loss_fn = nn.HuberLoss()

        return {
            'reorder': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def step(self, batch: T, mode='train'):

        embeddings = batch['embeddings']
        N, L, D = embeddings.shape

        X = np.arange(N)[None].T.repeat(L, axis=-1)
        Y = np.array([np.random.permutation(L) for _ in range(N)])
        shuffled_embs = embeddings[X, Y]

        scores = self.train_vars['reorder'](shuffled_embs)
        labels = torch.tensor(
            Y, 
            dtype=torch.float32, 
            device=scores.device,
        ) 

        loss = self.loss_fn(
            scores,
            labels,
        )

        """
        preds = torch.argsort(scores)
        tau, p_tau = kendalltau(Y, preds.cpu().numpy())
        rho, p_rho = spearmanr(
            Y.flatten(), 
            preds.cpu().numpy().flatten()
        )
        """
        scores = scores.detach().cpu().numpy()
        kendall = [
            tuple(kendalltau(y, score))
            for y, score in zip(Y, scores)
            #tuple(kendalltau(y[msk], pred))
            #for y, msk, pred in zip(Y, mask, preds)
        ]
        tau, p_tau = zip(*kendall)
        tau = np.mean(tau)
        p_tau = np.mean(p_tau)

        spearman = [
            tuple(spearmanr(y, score))
            for y, score in zip(Y, scores)
            #tuple(spearmanr(y[msk], pred))
            #for y, msk, pred in zip(Y, mask, preds)
        ]
        rho, p_rho = zip(*spearman)
        rho = np.mean(rho)
        p_rho = np.mean(p_rho)

        return loss, tau, rho, p_tau, p_rho

    def train_step(self, batch: T) -> Tuple[torch.Tensor, Dict]:
        """
        This function is used to compute the loss and training 
        statistics

        Input
            model[nn.Module]: this is the model you previously assigned.
            batch[T]: a batch sampled from the previously assigned train 
                dataloader

        Return
            loss[torch.Tensor]: the computed loss
            metrics[Dict]: the metrics you would like to track.
                refer to {project_root}/mltoolkit/trainers/base.py for 
                the currently supported keyword trackers
        """

        loss, tau, rho, p_tau, p_rho = self.step(batch, mode='train')

        return loss, {
            'scalar' : {
                'loss' : loss,
                'tau': tau,
                'rho': rho,
                'p_tau': p_tau,
                'p_rho': p_rho,
            }
        }

    def eval_step(self, batch: T, loader_name):
        """
        same as train_step but batch comes from previously assigned 
        val_loader (test_loader evaluation not implemented yet)
        NOTE: in this function, torch.Tensor values need to be converted 
        to float/int 

        Input
            batch[T]: a batch sampled from the previously assigned train 
                dataloader
            mode[str]: string that will be either 'val' or 'train' 
            depending on the procedure. 

        Return
            metrics[Dict]: the metrics you would like to track. Each run 
                of eval_step() will accumulate these metrics into a 
                Dataset object and will be used as input to 
                on_eval_end() for final aggregation
        """
        #loss, accuracy = self.step(batch, mode='val')

        loss, tau, rho, p_tau, p_rho = self.step(batch, mode='val')

        return {
            'loss': float(loss),
            'tau': float(tau),
            'rho': float(rho),
            'p_tau': float(p_tau),
            'p_rho': float(p_rho),
        }

    def on_eval_end(self, metrics: List, mode: str):
        """
        use this to aggregate your metrics collected from the outputs of 
        eval_step()

        Input:
            metrics[Dict[str, list]]: the set of metrics collected from 
                eval_step()

        Return
            target_metric[T]: the metric that will be used to determine 
                if the current model should be checkpointed to 
                {cfg.params['results']}/best_model.pt.
            log_values[Dict[Dict[str, T]]]: 
        """

        metrics_ds = Dataset.from_list(metrics['val_loader'])
        loss = np.mean(metrics_ds['loss'])
        tau = np.mean(metrics_ds['tau'])
        rho = np.mean(metrics_ds['rho'])
        p_tau = np.mean(metrics_ds['p_tau'])
        p_rho = np.mean(metrics_ds['p_rho'])

        return rho, {
            'scalar' : {
                'loss' : loss,
                'tau': tau,
                'rho': rho,
                'p_tau': p_tau,
                'p_rho': p_rho,
            }
        }
