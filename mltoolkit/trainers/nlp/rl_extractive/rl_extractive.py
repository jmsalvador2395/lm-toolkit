"""
This is an implementation for extractive summarization using reinforcement learning
"""

""" external imports """
import datasets
import io
import numpy as np
import torch
from PIL import Image
from datasets import Dataset
from matplotlib import pyplot as plt
from rouge_score.rouge_scorer import RougeScorer
from torch import nn
from torch.distributions import Categorical
from transformers import AutoTokenizer

""" local imports """
# parent class for the trainer
from mltoolkit.trainers.base import TrainerBase

# task-specific modules
from . import (
    data, # handles the dataset and dataloaders
    plotter # handles the plotting for tensorboard
)

# models
from mltoolkit.models import RLModel
from mltoolkit.models.pretrained import RewardModel

# utilities
from mltoolkit.utils import (
    files,
    strings,
    display,
    tokenizers,
)

class TrainerRLExtractive(TrainerBase):
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug=debug)

    def init_model(self):

        cfg = self.cfg

        # initialize model
        model = RLModel(cfg.model).to(self.dev)

        # initialize reward computation model
        self.reward_model = \
            RewardModel(cfg.model).to(cfg.model['reward_device'])
        self.reward_model.eval()

        return model

    def init_optimizer(self, model):

        cfg = self.cfg

        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
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

        # initialize tokenizer
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.data.get('tokenizer_name', 'bert-base-uncased'),
            do_lower_case=True,
        )

        # retrieve dataloaders
        train_loader, val_loader = \
            data.fetch_dataloaders(
                cfg,
                self.tokenizer
            )
        return train_loader, val_loader, None

    def init_aux(self):

        cfg = self.cfg

        self.k = cfg.model['budget']
        self.lmda = cfg.model['lambda']

        self.reward_hist = {
            'train': [],
            'val': []
        }
        self.rouge_hist = {
            'train': [],
            'val': []
        }

        self.step_counter = 0
        

    def evaluate_rouge(self, references, preds, sent_mat):
        scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        masks = preds.to(torch.bool).cpu().numpy()

        predictions = [
            ' '.join(sentences[mask]) 
            for sentences, mask 
            in zip(sent_mat, masks)
        ]

        # create huggingface dataset from rouge scores.
        # scores[<metric>] returns numpy array of shape (N, 3). columns are ['precision', 'recall', 'f-measeure']
        scores = Dataset.from_list([
            scorer.score(ref, pred)
            for ref, pred in zip(references, predictions)
        ]).with_format('numpy')

        category = ['precision', 'recall', 'fmeasure']

        # initialize dictionary of empty dictionaries
        metrics = {}

        # compute means for each metric
        for metric in scores.features.keys():
            for col, cat in enumerate(category):
                metrics[f'{metric}-{cat}'] = np.mean(scores[metric][:, col])
        metrics['step'] = self.step_counter

        return metrics

    def step(self, model, batch, mode='train'):

        # pop the reference doc
        tgt_txt = batch.pop('tgt_txt')

        # move inputs to device
        for doc in batch:
            for key, val in batch[doc].items():
                batch[doc][key] = \
                    val.to(self.dev) \
                    if type(val) == torch.Tensor \
                    else val

        # compute scores
        scores, mask_cls, shuffled_grid, unshuffled_grid = \
            model(batch)

        # sample actions
        preds = \
            Categorical(torch.exp(scores)).sample()
        preds *= mask_cls

        # get probabilities that correspond to each prediction
        row, col = np.indices(preds.shape)
        pred_probs = scores[row, col, preds]

        # compute base reward term
        sent_mat, base_term = self.reward_model(batch, preds)
        base_term = base_term.to(self.dev)

        # compute budget term
        sent_counts = torch.sum(preds, dim=-1)
        episode_lengths= torch.sum(mask_cls, dim=-1)
        budget_term = -abs(sent_counts - self.k) / episode_lengths

        # aggregate terms
        aggregate_reward = base_term + self.lmda*budget_term

        # save reward terms for plotting
        reward_parts = {
            'step': self.step_counter,
            'base term': torch.mean(base_term).detach().cpu().numpy(),
            'budget term': self.lmda*torch.mean(budget_term).detach().cpu().numpy(),
            'aggregate': torch.mean(aggregate_reward).detach().cpu().numpy(),
        }

        # compute loss
        loss = -pred_probs * aggregate_reward[:, None]
        loss = torch.mean(loss[mask_cls.bool()])
        
        # compute rouge
        rouge_scores = self.evaluate_rouge(
            tgt_txt, 
            preds,
            sent_mat
        )

        # use local step counter to decide when to compute rouge and append to history with reawrd
        if mode == 'train' and self.step_counter % self.cfg.data['log_freq'] == 0:

            self.reward_hist[mode].append(reward_parts)
            self.rouge_hist[mode].append(rouge_scores)

            rouge_plots = plotter.plot_rouge_history(
                self.rouge_hist[mode],
                mode
            )
            reward_plot = plotter.plot_reward_history(
                self.reward_hist[mode],
                mode
            )

            metrics = {
                'image': rouge_plots | reward_plot,
                'scalar': {
                    f'scalar/reward/{mode}': torch.mean(aggregate_reward).cpu().numpy(),
                    f'scalar/rouge1/{mode}': rouge_scores['rouge1-fmeasure'],
                    f'scalar/rouge2/{mode}': rouge_scores['rouge2-fmeasure'],
                    f'scalar/rougeL/{mode}': rouge_scores['rougeL-fmeasure'],
                },
                'histogram': {
                    f'reward base/{mode}': base_term.detach().cpu().numpy(),
                    f'reward budget/{mode}': budget_term.detach().cpu().numpy(),
                    f'reward aggregate/{mode}': aggregate_reward.detach().cpu().numpy(),
                },
            }
        elif mode == 'val':
            metrics = rouge_scores
            metrics.update({
                'loss': loss,
                'base term': base_term.detach().cpu().numpy(),
                'budget term': budget_term.detach().cpu().numpy(),
                'aggregate': aggregate_reward.detach().cpu().numpy(),
                'base term avg': reward_parts['base term'],
                'budget term avg': reward_parts['budget term'],
                'aggregate avg': reward_parts['aggregate'],
            })
        else:
            metrics = {}

        # increment step counter
        if mode == 'train': self.step_counter += 1

        return loss, metrics

    def train_step(self, model, batch):

        loss, metrics = self.step(model, batch)

        return loss, metrics

    def eval_step(self, model, batch, mode='val'):
        
        _, metrics = self.step(model, batch, mode=mode)

        return metrics

    def on_eval_end(self, aggregate_metrics, mode='val'):

        aggregate_metrics = aggregate_metrics.with_format('numpy')
        
        self.reward_hist[mode].append({
            'step': self.step_counter,
            'base term': np.mean(aggregate_metrics['base term avg']),
            'budget term': np.mean(aggregate_metrics['budget term avg']),
            'aggregate': np.mean(aggregate_metrics['aggregate avg']),
        })

        self.rouge_hist[mode].append({
            'step': self.step_counter,
            'rouge1-precision': np.mean(aggregate_metrics['rouge1-precision']),
            'rouge1-recall': np.mean(aggregate_metrics['rouge1-recall']),
            'rouge1-fmeasure': np.mean(aggregate_metrics['rouge1-fmeasure']),
            'rouge2-precision': np.mean(aggregate_metrics['rouge2-precision']),
            'rouge2-recall': np.mean(aggregate_metrics['rouge2-recall']),
            'rouge2-fmeasure': np.mean(aggregate_metrics['rouge2-fmeasure']),
            'rougeL-precision': np.mean(aggregate_metrics['rougeL-precision']),
            'rougeL-recall': np.mean(aggregate_metrics['rougeL-recall']),
            'rougeL-fmeasure': np.mean(aggregate_metrics['rougeL-fmeasure']),
        })

        rouge_plots = plotter.plot_rouge_history(
            self.rouge_hist[mode],
            mode
        )
        reward_plot = plotter.plot_reward_history(
            self.reward_hist[mode],
            mode
        )
        
        metrics = {
            'image': rouge_plots | reward_plot,
            'scalar': {
                f'loss/{mode}': np.mean(aggregate_metrics['loss']),
                f'scalar/reward/{mode}': np.mean(aggregate_metrics['aggregate avg']),
                f'scalar/rouge1/{mode}': self.rouge_hist[mode][-1]['rouge1-fmeasure'],
                f'scalar/rouge2/{mode}': self.rouge_hist[mode][-1]['rouge2-fmeasure'],
                f'scalar/rougeL/{mode}': self.rouge_hist[mode][-1]['rougeL-fmeasure'],
            },
            'histogram': {
                f'reward base/{mode}': np.concatenate([
                    np.array(arr)
                    for arr in aggregate_metrics['base term']
                ]),
                f'reward budget/{mode}': np.concatenate([
                    np.array(arr)
                    for arr in aggregate_metrics['budget term']
                ]),
                f'reward aggregate/{mode}': np.concatenate([
                    np.array(arr)
                    for arr in aggregate_metrics['aggregate']
                ]),
            },
        }

        reward = self.reward_hist[mode][-1]['aggregate']
        return reward, metrics

