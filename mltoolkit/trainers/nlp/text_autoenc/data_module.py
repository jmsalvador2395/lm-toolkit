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
import psutil
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from transformers import (
    AutoTokenizer,
)

# internal imports
from mltoolkit.utils import (
    display,
    files,
)

def fetch_dataloaders(cfg):

    ######################## set variables from cfg #######################

    train_batch_size = cfg.data['train_batch_size']
    val_batch_size = cfg.data['val_batch_size']
    shuffle = cfg.data.get('shuffle', True)
    num_workers = cfg.data.get('num_proc', 0)
    pin_memory = cfg.data.get('pin_memory', False)
    ds_save_dir = cfg.data.get('ds_save_dir', None)
    tokenizer = AutoTokenizer.from_pretrained(cfg.data['tokenizer_name'])
    max_seq_len = cfg.data['max_seq_len']
    seed = cfg.general.get('seed', None)

    ######################################################################

    if ds_save_dir is None:
        display.error('data/ds_save_dir not present in config file')
        raise ValueError()


    ds1 = datasets.load_dataset(
        'bookcorpus',
        cache_dir=cfg.data['cache_dir'],
        num_proc=num_workers
    )
    ds1 = ds1['train'].train_test_split(
        train_size=cfg.data['train_test_split'],
        seed=seed,
    )
    ds = ds1

    """
    ds2 = datasets.load_dataset(
        'wikitext',
        'wikitext-103-v1',
        cache_dir=cfg.data['cache_dir'],
        #num_proc=num_workers
    )

    # combine datasets
    ds = datasets.DatasetDict({
        'train': datasets.concatenate_datasets((ds1['train'], ds2['train'])),
        'test': datasets.concatenate_datasets((ds1['test'], ds2['test']))
    })

    # free other datasets
    del ds1
    del ds2

    sent_tokenizer = PunktSentenceTokenizer()
    ds = ds.map(
        token_map_fn,
        batched=True,
        fn_kwargs = {
            'sent_tokenizer': sent_tokenizer,
            'tokenizer': tokenizer,
            'seq_len': max_seq_len,
        },
        remove_columns=['text'],
        #num_proc=num_workers,
    )
    ds = ds.with_format('torch')
    """

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
            #train_data,
            ds['train'],
            batch_size=train_batch_size,
            #collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        DataLoader(
            #test_data,
            ds['test'],
            batch_size=val_batch_size,
            #collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        )
    )

def token_map_fn(batch, **kwargs):

    if '' in batch['text']:
        batch['text'] = list(filter(lambda text: text != '', batch['text']))
        if batch['text'] == []:
            return {'input_ids': [], 'attention_mask': []}

    # split sentences and filter out empty sentences
    sentences = list(itertools.chain.from_iterable(
        [sent_tokenize(doc) for doc in batch['text']]
    ))

    tokens = kwargs['tokenizer'](
        sentences,
        max_length=kwargs['seq_len']+1,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

    non_empty = np.sum(tokens['attention_mask'], axis=-1) != 2

    # filter out empty sentences
    if np.any(non_empty == False):

        display.warning('empty sentence found')

        tokens['input_ids'] = tokens['input_ids'][non_empty]
        tokens['attention_mask'] = tokens['attention_mask'][non_empty]

    return tokens

def collate_fn(batch):

    input_ids = torch.vstack([sample['input_ids'] for sample in batch])
    attention_mask = torch.vstack([sample['attention_mask'] for sample in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
