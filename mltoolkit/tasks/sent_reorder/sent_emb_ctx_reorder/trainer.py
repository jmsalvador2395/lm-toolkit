"""
this is an example trainer class that inherits TrainerBase
"""
# external imports
import torch
import numpy as np
import itertools
from torch import nn
from torch.nn import functional as F
from datasets import Dataset
from scipy.stats import kendalltau, spearmanr
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize
from torch import Tensor

# for typing
from typing import Tuple, List, Dict, TypeVar
T = TypeVar('T')

# local imports
from mltoolkit.templates import Trainer
from mltoolkit.utils import tensor_utils
from mltoolkit.models.sent_encoders.hf_models import from_hf
from mltoolkit.utils import (
    files,
    strings,
    display,
)
from .model import SentEmbedReorder, SentEmbedReorderCls
from .data_module import get_dataloaders
from .loss_functions import (
    hinge_loss, cross_entropy_loss, huber_loss,
    diff_kendall, hinge_pair_loss, exclusive
)

class TrainerSentEmbedCtxReordering(Trainer):
    def __init__(self, config_path, debug=False, accelerator=None):
        super().__init__(
            config_path,
            debug,
            accelerator=accelerator
        )

    def setup(self):
        cfg = self.cfg

        train_loader, val_loader = get_dataloaders(cfg)

        # define models
        model_name = cfg.params['encoder']
        self.method = cfg.params.get('method', 'regression')
        self.model_name = model_name
        encoder = AutoModel.from_pretrained(
            model_name,
            cache_dir=cfg.paths['cache'],
        )

        # freeze encoder if specified
        if cfg.params['freeze_encoder']:
            encoder.eval()

        if self.method == 'regression':
            model = SentEmbedReorder(**cfg.params)
        elif self.method == 'classification':
            model = SentEmbedReorderCls(**cfg.params)
        else:
            raise ValueError(
                '`method` parameter should be either `regression or '
                '`classification`.'
            )

        tok = AutoTokenizer.from_pretrained(model_name)

        # optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.params['lr'],
            weight_decay=self.cfg.params['weight_decay']
        )

        enc_optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=self.cfg.params['encoder_lr'],
            weight_decay=self.cfg.params['encoder_weight_decay'],
        )

        # scheduler
        lr_sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.params['sched_step_size'],
            gamma=cfg.params['sched_gamma'],
        )
        enc_sched = torch.optim.lr_scheduler.StepLR(
            enc_optimizer,
            cfg.params['sched_step_size'],
            gamma=cfg.params['sched_gamma'],
        )


        self.loss_fn_name = cfg.params.get('loss_fn', 'hinge')
        match self.loss_fn_name:
            case 'hinge':
                self.loss_fn = hinge_loss
            case 'cross_entropy':
                self.loss_fn = cross_entropy_loss
            case 'diff_kendall':
                self.loss_fn = diff_kendall
            case 'huber':
                self.loss_fn = huber_loss
            case 'hinge_pair':
                self.loss_fn = hinge_pair_loss
            case 'exclusive':
                self.loss_fn = exclusive
            case _:
                raise ValueError(
                    'loss function should be one of: [`hinge`, '
                    '`cross_entropy, `diff_kendall`, `huber`, '
                    '`hinge_pair`, `exclusive`'
                )


        return {
            'reorder': model,
            'encoder': encoder,
            'tok': tok,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'enc_optimizer': enc_optimizer,
            'optimizer': optimizer,
            'scheduler': lr_sched,
            'enc_scheduler': enc_sched,
        }

    def step(self, batch: T, mode='train'):

        tok = self.train_vars['tok']
        seq_len = self.cfg.params['seq_len']

        # convert docs to batch of sentences
        # NOTE: this also truncates each document down to 
        #       `seq_len` sentences
        sent_batch = [
            sents[:seq_len] for sents in batch['sentences']
        ]

        # get number of sentences per doc
        sent_lengths = [len(doc) for doc in sent_batch]

        #sent_batch = [text.split('<eos>') for text in batch['text']]
        N, L = len(sent_batch), max(sent_lengths)

        # create indexing arrays for shuffling
        X = np.arange(N)[None].T.repeat(L, axis=-1)
        Y = np.arange(L)[None].repeat(N, axis=0)

        # shuffle the indices while ignoring the padded dimensions
        for row, length in zip(range(N), sent_lengths):
            np.random.shuffle(Y[row, :length])
        
        # shuffle sentences
        sent_batch_shuf = [
            np.array(sents)[y[:L]].tolist()
            for sents, y, l in zip(sent_batch, Y, sent_lengths)
        ]

        if self.cfg.params['encoder'] == 'facebook/bart-large':
            bos_tok = tok.cls_token
            eos_tok = tok.eos_token
            sent_batch_cat = [
                f'{eos_tok}{bos_tok}'.join(sents)
                for sents in sent_batch_shuf
            ]
        else:
            raise ValueError('cfg->encoder value is invalid')

        # tokenize
        tokens = tok(
                sent_batch_cat,
                truncation=True,
                max_length=self.cfg.params['sent_len'],
                return_token_type_ids=False,
                padding=True,
                return_tensors='pt'
        ).to(self.accel.device)

        if self.cfg.params['encoder'] == 'facebook/bart-large':
            cls_ids = [
                torch.where(ids == tok.cls_token_id)[0]
                for ids in tokens['input_ids']
            ]

        # get sentence embeddings
        if self.cfg.params['freeze_encoder']:
            with torch.no_grad():
                word_embeds = \
                    self.train_vars['encoder'](**tokens)
        else:
            try:
                word_embeds = \
                    self.train_vars['encoder'](**tokens)
            except Exception as e:
                breakpoint()
        
        # group sentence embeddings by document
        embeds_per_doc = [
            embeds[ids] for embeds, ids 
            in zip(word_embeds['last_hidden_state'], cls_ids)
        ]
        embeds_per_doc = tensor_utils.pad_and_stack(embeds_per_doc)

        # create padding mask for embeds_per_doc
        embed_pad_mask = tensor_utils.get_pad_mask(
            embeds_per_doc.shape[:-1],
            np.array(sent_lengths),
            self.accel.device,
        )

        # compute scores
        scores = self.train_vars['reorder'](
            embeds_per_doc, embed_pad_mask
        )

        Xpt = torch.tensor(X, device=self.accel.device)
        Ypt = torch.tensor(Y, device=self.accel.device)
        label_mask = ~embed_pad_mask

        # compute loss
        loss, ordering = self.loss_fn(
            scores, Xpt, Ypt, label_mask,
            **self.cfg.params.get('loss_args')
        )

        # convert tensors to numpy
        ordering = ordering.cpu().numpy()
        label_mask = label_mask.cpu().numpy()

        kendall = [
            tuple(kendalltau(y[mask], order[mask]))
            for y, order, mask in zip(Y, ordering, label_mask)
        ]
        tau, p_tau = zip(*kendall)
        tau = np.mean(tau)
        p_tau = np.mean(p_tau)

        return loss, tau, p_tau

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

        loss, tau, p_tau = self.step(batch, mode='train')

        return loss, {
            'scalar' : {
                'loss' : loss,
                'tau': tau,
                'p_tau': p_tau,
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

        loss, tau, p_tau, = self.step(batch, mode='val')

        return {
            'loss': float(loss),
            'tau': float(tau),
            'p_tau': float(p_tau),
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
        p_tau = np.mean(metrics_ds['p_tau'])

        return tau, {
            'scalar' : {
                'loss' : loss,
                'tau': tau,
                'p_tau': p_tau,
            }
        }
