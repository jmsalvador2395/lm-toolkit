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
    reward_module,
)

# models
from mltoolkit.models import RLModel, SofsatExtractor, TransformerDecoder

# utilities
from mltoolkit.utils import (
    files,
    strings,
    display,
    tokenizers,
)

class TrainerSofsatRankingDCG(TrainerBase):
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug=debug)

    def init_model(self):

        cfg = self.cfg

        #model = SofsatExtractor(cfg).to(self.dev)
        model = TransformerDecoder(cfg).to(self.dev)

        # initialize reward computation model
        self.reward_computer = \
            reward_module.RewardModule(cfg).to(cfg.model['reward_device'])
        self.reward_computer.eval()

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

        self.k = cfg.model.get('budget', 3)
        self.A = cfg.model['out_size']

        self.reward_hist = {
            'train': [],
            'val': []
        }
        self.rouge_hist = {
            'train': [],
            'val': []
        }

        self.step_counter = 0
        

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

    def step(self, model, batch, mode='train'):
        
        ######### unpack bach dictionary ######### 

        seq = batch.pop('sequence')
        seq_mask = batch['seq_mask']
        sep_mask = batch['sep_mask']
        N, S = seq_mask.shape
        row, col = np.indices(seq_mask.shape)

        ##########################################

        # make predictions
        pad_mask = ~(batch['seq_mask'] | batch['sep_mask'])
        dist = model(seq, pad_mask)

        if mode == 'train':
            rankings = dist.sample()
        else:
            rankings = torch.argmax(dist.probs, dim=-1)
        #rankings = dist.sample()

        log_probs = dist.log_prob(rankings)

        # evaluate the episode to get reward, metrics and predicted documents
        reward, doc_preds = self.reward_computer(
            batch,
            rankings,
        )

        # compute loss
        #loss = -log_probs * (reward[:, None] * 10)
        loss = -log_probs * reward[:, None]
        loss = torch.mean(loss[seq_mask])# - .01*torch.mean(dist.entropy())

        # compute rouge
        rouge_scores = self.evaluate_rouge(
            batch['intersection'],
            doc_preds,
        )

        """
        if self.step_counter == 200:
            breakpoint()
        """

        # udpate metrics
        metrics = {'step': self.step_counter}
        metrics.update(rouge_scores)
        metrics.update({
            'loss': loss,
            'action distribution': torch.flatten(rankings),
            'target docs': batch['intersection'],
            'predicted docs': doc_preds,
            'reward': reward,
            'average reward': torch.mean(reward),
        })

        return loss, metrics

    def train_step(self, model, batch):

        loss, metrics = self.step(model, batch)
        mode = 'train'

        if self.step_counter % self.cfg.data['log_freq'] == 0:

            # update rouge history and plot 
            rouge_metrics = {
                key[6:]: val for key, val in metrics.items()
                if 'rouge' in key
            }
            rouge_metrics['step'] = self.step_counter
            self.rouge_hist[mode].append(rouge_metrics)

            rouge_plots = plotter.plot_rouge_history(
                self.rouge_hist[mode],
                mode
            )

            # collect histogram metrics (reward terms, action probs, and extracted sentences)
            histogram_metrics = {
                key: val for key, val in metrics.items()
                if 'reward' in key
            }
            avg_reward = histogram_metrics.pop('average reward')
            histogram_metrics.update({
                'action distribution': metrics['action distribution'],
            })

            # build categorized metric dictionary
            categorized_metrics = {
                'image': rouge_plots,
                'histogram': histogram_metrics,
                'text': md_table(
                    metrics['target docs'],
                    metrics['predicted docs'] 
                ),
                'scalar': {
                    'loss': metrics['loss'],
                    'rouge/rouge1': metrics['rouge/rouge1-fmeasure'],
                    'rouge/rouge2': metrics['rouge/rouge2-fmeasure'],
                    'rouge/rougeL': metrics['rouge/rougeL-fmeasure'],
                    'average reward': avg_reward,
                },
            }

            metrics = categorized_metrics
        else:
            metrics = {}

        self.step_counter += 1

        return loss, metrics

    def eval_step(self, model, batch, mode='val'):
        
        _, metrics = self.step(model, batch, mode=mode)

        return metrics

    def on_eval_end(self, ag_metrics, mode='val'):

        ag_metrics = ag_metrics.with_format('numpy')
        am_keys = ag_metrics.features.keys()
        
        # log rouge metrics and plot
        rouge_metrics = {
            key[6:]: np.mean(ag_metrics[key]) for key
            in am_keys if 'rouge' in key
        }
        rouge_metrics['step'] = self.step_counter
        self.rouge_hist[mode].append(rouge_metrics)

        rouge_plots = plotter.plot_rouge_history(
            self.rouge_hist[mode],
            mode
        )

        # group histogram metrics
        histogram_keys = [key for key in am_keys if 'reward' in key]
        histogram_keys.pop(histogram_keys.index('average reward'))
        histogram_keys += [
            'action distribution',
        ]

        histogram_metrics = {
            key: np.concatenate(ag_metrics[key]) 
            for key in am_keys if key in histogram_keys
        }

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

        reward = np.mean(metrics['histogram']['reward'])
        metrics['scalar']['average reward'] = reward

        return reward, metrics

def md_table(targets, preds):

    # build header
    table = (
          '# Target Intersections Vs. Predictions\n'
        + '\n'
        + '| ID | Target | Prediction |\n'
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
