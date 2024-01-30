"""
this file contains code for changing the All-Sides dataset to fit the RL Extractive Summarization task
"""

# external imports
import random
import datasets
import os
import torch
import numpy as np
import itertools
from math import comb
from itertools import combinations
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    AutoTokenizer,
)
from tqdm import tqdm

# internal imports
from mltoolkit.utils import display

def fetch_dataloaders(cfg):

    ######################## set variables from cfg #######################

    train_batch_size = cfg.data['train_batch_size']
    val_batch_size = cfg.data['val_batch_size']
    shuffle = cfg.data.get('shuffle', True)
    num_workers = cfg.data.get('num_proc', 0)
    pin_memory = cfg.data.get('pin_memory', False)
    overlap_data_path = cfg.data.get('overlap_data_path', None)

    ######################################################################

    # first read in data as a huggingface dataset
    ds1 = datasets.load_dataset(
        'wikipedia',
        '20220301.en',
        cache_dir=cfg.data['cache_dir'],
        num_proc=num_workers
    )

    ds2 = datasets.load_dataset(
        'snli',
        cache_dir=cfg.data['cache_dir'],
        num_proc=num_workers,
    )

    overlap_ds = datasets.Dataset.from_csv(
        overlap_data_path,
        cache_dir=cfg.data['cache_dir'],
        num_proc=num_workers
    )

    
    # break wikipedia dataset into sentences
    ds1 = ds1.map(
        wiki_map_fn,
        batched=True,
        batch_size=1,
        remove_columns=ds1.column_names['train'],
        num_proc=num_workers
    )

    # put all text in dataset into single "text" column
    ds2 = ds2.map(
        snli_map_fn,
        batched=True,
        batch_size=1,
        remove_columns=ds2.column_names['train'],
        num_proc=num_workers
    )

    # combine wikipedia and snli datasets
    ds = datasets.concatenate_datasets([
        ds1['train'],
        ds2['train'],
    ])
    ds = ds.train_test_split(train_size=cfg.data['train_test_split'])

    # convert to AllSides format.
    train_data, test_data = (
        TextData(ds['train'], cfg),
        TextData(ds['test'], cfg),
    )

    # define function for RNG in dataloaders
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(cfg.general['seed'])

    # return dataloader for train and test sets
    return (
        DataLoader(
            train_data,
            batch_size=train_batch_size,
            collate_fn=train_data.collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        DataLoader(
            overlap_ds,
            batch_size=val_batch_size,
            #collate_fn=test_data.collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        )
    )

def wiki_map_fn(batch):
    sents = sent_tokenize(batch['text'][0])
    return {'text': sents}

def snli_map_fn(batch):
    return {
        'text': [batch['premise'][0], batch['hypothesis'][0]]
    }


class TextData(Dataset):
    """
    this class is made to serve arbitrary text data in the pytorch format
    """
    
    def __init__(self, ds, cfg):

        ######################## set variables from cfg #######################

        self.name = 'Text Data'

        ######################################################################

        # assign dataset 
        self.ds = ds

        #self.compute_statistics()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def collate_fn(self, batch: List[Tuple]):

        """
        sample_batch = [sample['sentences'] for sample in batch]
        sample_batch = list(itertools.chain.from_iterable(sample_batch))
        if len(sample_batch) > self.sample_limit:
            sample_batch = random.sample(sample_batch, self.sample_limit)
        """
        sent_batch = [sample['text'] for sample in batch]

        return sent_batch
