"""
This is an implementation for extractive summarization using reinforcement learning
"""

""" external imports """
import datasets
import io
import numpy as np
import os
import torch

from PIL import Image
from datasets import Dataset
from matplotlib import pyplot as plt
from rouge_score.rouge_scorer import RougeScorer
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer

""" local imports """
# parent class for the trainer
from mltoolkit.trainers.base import TrainerBase

# task-specific modules
from . import (
    data_module, # handles the dataset and dataloaders
    plotter, # handles the plotting for tensorboard
)

# models
from mltoolkit.models import TextAutoencoder

# utilities
from mltoolkit.utils import (
    files,
    strings,
    display,
    tokenizers,
)

class TrainerTextAutoencoder(TrainerBase):
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug=debug)

    def init_model(self):

        cfg = self.cfg

        # need to initialize tokenizer before defining model due to vocabulary size being tied to output size
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.data['tokenizer_name'])

        # define model
        cfg.model['vocab_size'] = len(self.tokenizer)
        cfg.model['pad_token_id'] = self.tokenizer.pad_token_id
        """
        model = torch.nn.DataParallel(
            TextAutoencoder(cfg),
            cfg.model['devices']
        )
        """
        model = TextAutoencoder(cfg).to(cfg.model['devices'][0])
        torch.compile()

        return model

    def init_optimizer(self):

        cfg = self.cfg

        # optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optim['lr'],
            betas=(
                cfg.optim.get('beta1', .9),
                cfg.optim.get('beta2', .999)
            ),
            eps=float(cfg.optim.get('eps', 1e-8)),
            weight_decay=float(self.cfg.optim.get('weight_decay', 1e-4)),
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.optim['sched_step_size'],
            gamma=cfg.optim['sched_gamma'],
        )
        return optimizer, scheduler

    def init_data(self):

        cfg = self.cfg

        # retrieve dataloaders
        train_loader, val_loader = \
            data_module.fetch_dataloaders(cfg)

        return train_loader, val_loader, None

    def init_aux(self):

        cfg = self.cfg

        # define devices to be used
        self.devs = cfg.model['devices']

        self.src_sents = []
        self.pred_sents = []

        self.inference_flag = True

    def train_step(self, model, batch):
        return self.loss_step(model, batch)

    def eval_step(self, model, batch, mode='val'):

        metrics = {}
        """
        if self.inference_flag:
            metrics.update(self.inference_step(model, batch))
            self.inference_flag = False
            breakpoint()
        """
        loss, loss_metrics = self.loss_step(model, batch, mode=mode)

        return {
            'loss': loss,
        }

    def on_eval_end(self, ag_metrics, mode='val'):

        """
        metrics = {
            'image': rouge_plots,
            'histogram': histogram_metrics,
            'text': md_table(
                list(np.concatenate(ag_metrics['target docs'])),
                list(np.concatenate(ag_metrics['predicted docs'])),
            ),
            'scalar': {
                'loss': np.mean(ag_metrics['loss']),
                'rouge/rouge1': self.rouge_hist[mode][-1]['rouge1-fmeasure'],
                'rouge/rouge2': self.rouge_hist[mode][-1]['rouge2-fmeasure'],
                'rouge/rougeL': self.rouge_hist[mode][-1]['rougeL-fmeasure'],
            },
        }
        """
        loss = np.mean(ag_metrics['loss'])

        metrics = {
            'scalar': {
                'loss': loss,
                'perplexity': 2**loss
            }
        }

        # clear saved samples
        self.src_sents = []
        self.pred_sents = []
        self.inference_flag = True

        return loss, metrics

    def loss_step(self, model, batch, mode='train'):

        ######## unpack vars ########

        cfg = self.cfg
        max_seq_len = cfg.data['max_seq_len']
        mask_prob = cfg.data.get('mask_prob', .01)

        #############################

        tokens = self.tokenizer(
            batch['text'],
            add_special_tokens=True,
            max_length=max_seq_len+1,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        tokens = tokens.to(self.devs[0])

        input_ids = tokens['input_ids']
        attn_mask = tokens['attention_mask']


        """
        pad_mask = ~pad_mask.to(torch.bool)

        src_ids, src_mask = (
            input_ids[:, 1:].clone(),
            pad_mask[:, 1:].clone()
        )
        src_mask[src_ids == self.tokenizer.eos_token_id] = True

        tgt_ids, tgt_mask = input_ids[:, :max_seq_len], pad_mask[:, :max_seq_len]
        label_ids, label_mask = input_ids[:, 1:], ~pad_mask[:, 1:]

        # compute forward pass
        scores = model(
            src_ids,
            src_mask,
            tgt_ids,
            tgt_mask
        )
        """

        src_ids, src_mask = (
            input_ids[:, :-1].clone(),
            attn_mask[:, :-1].clone()
        )

        label_ids, label_mask = (
            input_ids[:, 1:].clone(),
            attn_mask[:, 1:].clone()
        )
        label_mask = (label_mask != 0)

        scores = model(src_ids, src_mask, mask_prob)

        # compute loss
        loss_fn = nn.CrossEntropyLoss()

        N, S, O = scores.shape

        scores = scores[label_mask]
        label_ids = label_ids[label_mask]

        if torch.any(torch.isnan(scores)):
            breakpoint()

        # filter out nans from scores
        if torch.any(torch.isnan(scores)):
            display.warning(f'nan found at step {self.step}')
            nans = torch.any(torch.isnan(scores), dim=-1)
            scores = scores[~nans]
            label_ids = label_ids[~nans]

        loss = loss_fn(scores, label_ids)

        metrics = {
            #'image': rouge_plots,
            #'histogram': histogram_metrics,
            #'text': md_table(
            #    metrics['target docs'],
            #    metrics['predicted docs'] 
            #),
            'scalar': {
                'perplexity': 2**loss
            },
        }

        return loss, metrics
        
    def evaluate_rouge(self, references, preds):
        scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        # create huggingface dataset from rouge scores.
        # scores[<metric>] returns numpy array of shape (N, 3). columns are ['precision', 'recall', 'f-measeure']
        scores = Dataset.from_list([
            scorer.score(ref, pred)
            for ref, pred in zip(references, preds)
        ]).with_format('numpy')

        category = ['precision', 'recall', 'fmeasure']

        # initialize dictionary of empty dictionaries
        metrics = {}

        # compute means for each metric
        for metric in scores.features.keys():
            for col, cat in enumerate(category):
                metrics[f'rouge/{metric}-{cat}'] = np.mean(scores[metric][:, col])

        return metrics

def md_table(refs, decodings, groups):

    # build header
    table = (
          '# Intermediate Decodings on Overap Dataset\n'
        + '\n'
        + '| s1 | s2 | overlap | overlap AE | middle dec | middle*1.01 dec | middle*0.99 dec |\n'
        + '| - | - | - |\n'
    )

    # populate table
    for i in range(len(preds)):

        # format text for markdown
        trgt = targets[i].replace('\n', '<newline>').replace('|', '<vbar>')
        pred = preds[i].replace('\n', '<newline>').replace('|', '<vbar>')

        table += f'| {i} | {trgt} | {pred} |\n'

    return {
        'predicted documents': table
    }
