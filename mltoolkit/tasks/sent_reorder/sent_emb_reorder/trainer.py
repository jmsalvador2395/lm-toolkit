"""
this is an example trainer class that inherits TrainerBase
"""
# external imports
import torch
from torch import nn
import numpy as np
from datasets import Dataset
from scipy.stats import kendalltau, spearmanr
from nltk import sent_tokenize
from itertools import chain
from numpy.lib.stride_tricks import sliding_window_view

# for typing
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
from .model import SentEmbedReorder
from .data_module import get_dataloaders
from mltoolkit.models.sent_encoders.hf_models import from_hf

class TrainerSentEmbedReordering(Trainer):
    def __init__(self, config_path, debug=False, accelerator=None):
        super().__init__(
            config_path,
            debug,
            accelerator=accelerator
        )

    def setup(self):
        cfg = self.cfg

        train_vars = get_dataloaders(cfg)

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

        # sentence encoder
        model_name = 'mixedbread-ai/mxbai-embed-large-v1'
        encoder = from_hf(
            model_name, 
            emb_dim=1024, 
            max_seq_len=512,
            cache_dir=cfg.paths['cache'],
        )

        self.loss_fn = nn.HuberLoss()
        train_vars.update({
            'reorder': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'encoder': encoder,
        })

        return train_vars

    def step(self, batch: T, mode='train'):

        # prepare batch of sentences
        text = batch['text']
        sent_batch = [sent_tokenize(sent) for sent in text]
        lengths = np.array([len(sents) for sents in sent_batch])
        L = min(self.cfg.params['n_sents'], max(lengths))
        trunc_lengths = np.minimum(
            lengths,
            self.cfg.params['n_sents'],
        )
        diffs = np.maximum(0, L - lengths)

        #sent_batch = [sents[:L] for sents in sent_batch]
        sent_batch = list(map(lambda x: x[:L], sent_batch))
        sent_batch = list(chain.from_iterable(sent_batch))

        sent_embs = self.train_vars['encoder'].encode(
            sent_batch,
            convert_to_tensor=True,
            device=self.accel.device
        )
        indices = sliding_window_view(
            np.cumsum(np.hstack(([0], trunc_lengths))), 
            2
        )
        sent_embs = [
            sent_embs[start:stop]
            for start, stop in indices
        ]

        input_embs = tensor_utils.pad_and_stack(
            sent_embs,
            stack_dim=0,
            pad_dim=0,
        )
        N, L, E = input_embs.shape

        # create mask
        mask = torch.zeros(
            (N, L), 
            dtype=torch.long, 
            device=self.accel.device,
        )
        mask_trgts = diffs > 0
        mask[mask_trgts, lengths[mask_trgts]] = True
        mask = mask.cumsum(axis=-1).to(torch.bool)

        #X = np.arange(N)[None].T.repeat(L, axis=-1)
        #shuffled_embs = embeddings[X, Y]
        X = torch.arange(N)[..., None].repeat((1, L))
        Y = torch.arange(L).repeat(N, 1)
        for row, tl in zip(Y, trunc_lengths):
            row[:tl] = torch.tensor(np.random.permutation(tl))
        
        X, Y = X.to(self.accel.device), Y.to(self.accel.device)

        shuffled_embs = sent_embs[X, Y]

        scores = self.train_vars['reorder'](shuffled_embs)
        labels = 1 + torch.tensor(
            Y, 
            dtype=torch.float32, 
            device=scores.device,
        ) 

        loss = self.loss_fn(
            scores,
            labels,
        )

        preds = torch.argsort(scores)
        tau, p_tau = kendalltau(Y, preds.cpu().numpy())
        rho, p_rho = spearmanr(
            Y.flatten(), 
            preds.cpu().numpy().flatten()
        )

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
