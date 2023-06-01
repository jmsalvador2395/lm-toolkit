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
from torch.utils.data import DataLoader

# local imports
from mltoolkit.trainers.base import TrainerBase
from mltoolkit.utils import (
    files,
    strings,
    display,
)

class TrainerMNIST(TrainerBase):
    
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug)

    def init_data(self):

        cfg = self.cfg

        # load mnist dataset
        self.ds = datasets.load_dataset(
            'mnist',
            cache_dir=cfg.data['cache_dir']
        ).with_format('torch')
        in_size = cfg.model['hidden_dim']

        # preprocess
        def normalize(batch):
            batch.update({
                'image' : batch['image'].to(torch.float32)/255.
            })
            return batch

        ds = self.ds.map(
            normalize,
            batched=True,
            batch_size=1000,
            num_proc=cfg.data['num_proc'],
        )

        train_loader = DataLoader(
            ds['train'],
            shuffle=cfg.data['shuffle'],
            batch_size=cfg.data['batch_size'],
        )

        val_loader = DataLoader(
            ds['test'],
            shuffle=cfg.data['shuffle'],
            batch_size=cfg.data['batch_size'],
        )

        return train_loader, val_loader, None

    def init_model(self):
        cfg = self.cfg

        # define model
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                cfg.model['in_size'], 
                cfg.model['hidden_dim']
            ),
            torch.nn.BatchNorm1d(cfg.model['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.model['dropout']),
            torch.nn.Linear(
                cfg.model['hidden_dim'],
                cfg.model['num_classes'],
            )
        ).to(self.dev)

        return model

    def init_optimizer(self, model):

        cfg = self.cfg

        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.optim['lr'],
            weight_decay=self.cfg.optim['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.optim['sched_step_size'],
            gamma=cfg.optim['sched_gamma'],
        )
        
        return optimizer, scheduler

    def init_aux(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def step(self, model, batch, mode='train'):

        # compute scores and calculate loss
        scores = model(batch['image'].to(self.dev))
        labels = batch['label'].to(self.dev)
        loss = self.loss_fn(scores, labels)
        accuracy = torch.sum(torch.argmax(scores, dim=-1) == labels)/len(labels)

        return loss, accuracy

    def train_step(self, model, batch):

        loss, accuracy = self.step(model, batch, mode='train')

        return loss, {
            'scalar' : {
                'loss/train' : loss,
                'accuracy/train' : accuracy
            }
        }

    def eval_step(self, model, batch, mode='val'):
        loss, accuracy = self.step(model, batch, mode=mode)

        return {
            'loss': loss,
            'accuracy': accuracy,
        }

    def on_eval_end(self, metrics, mode='val'):

        loss = np.mean(metrics['loss'])
        accuracy = np.mean(metrics['accuracy'])

        return accuracy, {
            'scalar' : {
                'loss/val' : loss,
                'accuracy/val' : accuracy
            }
        }
