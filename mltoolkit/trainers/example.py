"""
this is an example trainer class that inherits TrainerBase
"""
# external imports
import torch
import numpy as np
import datasets
from tqdm import tqdm
from torch import nn
from datasets import Dataset

# local imports
from .base import TrainerBase
from mltoolkit.utils import (
    files,
    strings,
    display,
    data
)

class TrainerExample(TrainerBase):
    
    def __init__(self, config_path, debug=False):
        super().__init__(config_path)

        # define model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        ).to(self.dev)

        # load in datasets
        N, C = (4, 10)
        ds_train = Dataset.from_dict({
            'data' : np.random.randn(N, C),
            'labels' : np.random.randint(0, C, N)
        }).with_format('torch')

        ds_test = Dataset.from_dict({
            'data' : np.random.randn(N, C),
            'labels' : np.random.randint(0, C, N)
        }).with_format('torch')

        self.ds = datasets.DatasetDict({
            'train' : ds_train,
            'test' : ds_test
        })

    def init_optimizer(self):
        # optimizer
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.optim['lr'],
            weight_decay=self.cfg.optim['weight_decay']
        )

    def init_loss_fn(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def prepare_data_and_tools(self):
        # do your dataset mapping here
        ds = ds.map(lambda sample: sample)

        # set some other tools that you would need here
        example_var = 50

    def evaluate(self):
        return 0, {}

    def train_step(self, batch):
        # compute scores and calculate loss
        scores = self.model(batch['data'].to(self.dev))
        labels = batch['labels'].to(self.dev)
        loss = self.loss_fn(scores, labels)

        return loss, {}
