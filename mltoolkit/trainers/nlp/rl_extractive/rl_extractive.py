"""
This is an implementation for extractive summarization using reinforcement learning
"""

""" external imports """
import torch
import numpy as np
import datasets
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
from torch import nn
from torch.nn import functional as f
from torch.distributions import Categorical

""" local imports """
# parent class for the trainer
from mltoolkit.trainers.base import TrainerBase

# data prep uses a lot of code so its functions are consolidated into this file
from . import data

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
        self.model = RLModel(cfg.model).to(self.dev)

        # initialize reward computation model
        self.reward_model = RewardModel(
            cfg.model.get(
                'reward_model',
                'all-mpnet-base-v1'
            )
        )

    def init_optimizer(self):

        cfg = self.cfg

        # optimizer
        self.optimizer = torch.optim.Adam(
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
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            cfg.optim['sched_step_size'],
            gamma=cfg.optim['sched_gamma'],
        )

    def init_loss_fn(self):

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def init_data_and_misc(self):

        cfg = self.cfg

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.data.get('tokenizer_name', 'bert-base-uncased'),
            do_lower_case=True,
        )

        # retrieve dataloaders
        self.train_loader, self.val_loader = \
            data.fetch_dataloaders(
                cfg.data,
                self.tokenizer
            )

    def train_step(self, batch):

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
            self.model(batch)

        # sample actions
        preds = \
            Categorical(torch.exp(scores)).sample()

        # get probabilities that correspond to each prediction
        row, col = np.indices(preds.shape)
        pred_probs = scores[row, col, preds]

        breakpoint()
        return loss, {
            'scalar': {
                'loss/train': loss,
            }
        }
