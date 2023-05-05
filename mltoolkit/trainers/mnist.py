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

class TrainerMNIST(TrainerBase):
    
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug)

        self.C = 10
        cfg = self.cfg

        # load mnist dataset
        self. ds = datasets.load_dataset(
            'mnist',
            cache_dir=cfg.data['cache_dir']
        ).with_format('torch')
        in_size = cfg.model['hidden_dim']

        # define model
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                cfg.model['in_size'], 
                cfg.model['hidden_dim']
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                cfg.model['hidden_dim'],
                self.C
            )
        ).to(self.dev)

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
        def normalize(sample):
            sample.update({
                'image' : sample['image'].to(torch.float32)/255.
            })
            return sample
        self.ds = self.ds.map(normalize)

    def evaluate(self):
        with torch.no_grad():
            scores = self.model(self.ds['test'][:]['image'].to(self.dev))
            labels = self.ds['test'][:]['label'].to(self.dev)
            loss = self.loss_fn(scores, labels)

        accuracy = torch.sum(torch.argmax(scores, dim=-1) == labels)/len(labels)
        return accuracy, {
            'scalar' : {
                'loss/test' : loss,
                'accuracy/test' : accuracy
            }
        }

    def train_step(self, batch):
        # compute scores and calculate loss
        scores = self.model(batch['image'].to(self.dev))
        labels = batch['label'].to(self.dev)
        loss = self.loss_fn(scores, labels)
        accuracy = torch.sum(torch.argmax(scores, dim=-1) == labels)/len(labels)

        return loss, {
            'scalar' : {
                'loss/train' : loss,
                'accuracy/train' : accuracy
            }
        }
