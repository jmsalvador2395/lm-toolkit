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
    reward_model,
)

# models
from mltoolkit.models import RLModel, SofsatExtractor

# utilities
from mltoolkit.utils import (
    files,
    strings,
    display,
    tokenizers,
)

class TrainerSofsatExtractiveRL(TrainerBase):
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug=debug)

    def init_model(self):

        cfg = self.cfg

        model = SofsatExtractor(cfg).to(self.dev)

        # initialize reward computation model
        self.reward_model = \
            reward_model.RewardModel(cfg).to(cfg.model['reward_device'])
        self.reward_model.eval()

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

        if self.A not in [2, 4]:
            display.error(
                f'invalid out_size (action space): {self.A}. only 2 and 4 are valid'
            )
            raise ValueError()

        self.reward_hist = {
            'train': [],
            'val': []
        }
        self.rouge_hist = {
            'train': [],
            'val': []
        }

        self.step_counter = 0
        self.stay_in_budget = cfg.model.get('stay_in_budget', False)
        

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
        #display.debug('training step')
        ######### unpack bach dictionary ######### 

        seq = batch.pop('sequence')
        seq_mask = batch['seq_mask']
        N, S = seq_mask.shape
        row, col = np.indices(seq_mask.shape)

        ##########################################

        # make predictions
        scores = model(seq)
        """
        preds = Categorical(torch.exp(scores)).sample()

        # get probabilities that correspond to each prediction
        row, col = np.indices(preds.shape)
        log_probs = scores[row, col, preds]
        picked_indices = col[
            (preds.detach().bool() * seq_mask).cpu().numpy()
        ]
        """
        log_probs = scores[..., -1]
        masked_log_probs = log_probs.detach().clone()
        masked_log_probs[~seq_mask] = float('-inf')

        picked_indices = torch.argsort(
            masked_log_probs,
            dim=-1,
            descending=True
        )
        picked_indices = picked_indices[:, :self.k]
        preds = torch.zeros(
            seq_mask.shape,
            dtype=torch.bool,
            device=self.dev,
        )
        preds[row[:, :self.k], picked_indices] = True

        # evaluate the episode to get reward, metrics and predicted documents
        reward, metrics, doc_preds = \
            self.reward_model.evaluate_episodes(
                batch,
                preds,
                action_space=self.A,
                ver=self.cfg.model['reward_ver'],
            )

        # compute loss
        loss = -log_probs * reward[:, None]
        loss = torch.mean(loss[seq_mask])
        
        # compute rouge
        rouge_scores = self.evaluate_rouge(
            batch['intersection'],
            doc_preds,
        )

        # udpate metrics
        metrics['step'] = self.step_counter
        metrics.update(rouge_scores)
        metrics.update({
            'loss': loss,
            'action probs': torch.exp(log_probs[seq_mask]).detach().cpu().numpy(),
            'extracted sentences': picked_indices,
            'extracted sentences (normalized)': \
                picked_indices/torch.sum(
                    seq_mask,
                    dim=-1,
                    keepdim=True
                ),
            'target docs': batch['intersection'],
            'predicted docs': doc_preds,
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
            histogram_metrics.update({
                'action probs': metrics['action probs'],
                'extracted sentences': metrics['extracted sentences'],
                'extracted sentences (normalized)': metrics['extracted sentences (normalized)'],
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
                    'reward avg/aggregate': np.mean(metrics['reward/aggregate']),
                    'rouge/rouge1': metrics['rouge/rouge1-fmeasure'],
                    'rouge/rouge2': metrics['rouge/rouge2-fmeasure'],
                    'rouge/rougeL': metrics['rouge/rougeL-fmeasure'],
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
        histogram_keys += [
            'episode lengths',
            'action probs',
            'extracted sentences',
            'extracted sentences (normalized)',
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
                'reward avg/aggregate': np.mean(
                    np.concatenate(ag_metrics['reward/aggregate']
                )),
                'rouge/rouge1': self.rouge_hist[mode][-1]['rouge1-fmeasure'],
                'rouge/rouge2': self.rouge_hist[mode][-1]['rouge2-fmeasure'],
                'rouge/rougeL': self.rouge_hist[mode][-1]['rougeL-fmeasure'],
            },
        }

        reward = metrics['scalar']['reward avg/aggregate']
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
