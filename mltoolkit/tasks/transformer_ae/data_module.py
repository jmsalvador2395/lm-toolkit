# this module is for downloading and combining the bookcorpus and wikipedia datasets and preparing them for 
# MLM and NSP

import datasets
import torch
import numpy as np
import random
from nltk import sent_tokenize
from torch.utils.data import DataLoader

from mltoolkit.utils import tensor_utils

def get_dataloaders(cfg):
    """
    retrieves dataset and converts to dataloaders
    """

    trgt_dir = cfg.paths['cache'] + '/pg19_plus_wiki'
    try:
        ds = datasets.load_from_disk(trgt_dir)
        ds_loaded = True
    except Exception as e:
        ds_loaded = False

    if not ds_loaded:
        ds = combine_datasets(cfg)
        ds.save_to_disk(trgt_dir)

    seed_worker, g = tensor_utils.get_dl_params(cfg.general['seed'])

    train_loader = DataLoader(
        ds['train'],
        batch_size=cfg.params['batch_size'],
        num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        ds['validation'],
        batch_size=cfg.params['batch_size'],
        num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader

def combine_datasets(cfg):
    """
    This function performs the following:
        - loads in and then combines the pg19 and wikipedia datasets
        - splits each document by sentences, 
        - tags the last sentence in each document
    """

    # for extracting test samples from wikipedia
    num_test = cfg.params['num_test_samples']

    # load pg19 dataset
    pg19 = datasets.load_dataset(
        'pg19',
        cache_dir=cfg.paths['cache'],
        trust_remote_code=True,
    )
    pg19 = pg19.remove_columns([
        'short_book_title', 
        'publication_date',
        'url',
    ])

    wiki = datasets.load_dataset(
        'wikipedia',
        '20220301.en',
        cache_dir=cfg.paths['cache'],
        trust_remote_code=True,
    )

    wiki = wiki.remove_columns(['id', 'url', 'title'])
    len_wiki = len(wiki['train'])

    train_ds = datasets.concatenate_datasets([
        pg19['train'],
        pg19['validation'],
        wiki['train'].select(range(len_wiki-num_test)),
    ])


    val_ds = datasets.concatenate_datasets([
        pg19['test'].select(range(num_test)),
        wiki['train'].select(range(len_wiki-num_test, len_wiki)),
    ])

    ds = datasets.DatasetDict({
        'train': train_ds,
        'validation': val_ds,
    })
    
    # mapping function
    def split_sentences(in_batch, indices):
        out_batch = {
            'text': [],
            'doc_id': [],
            'last_sent_flag': [],
        }

        for doc, idx in zip(in_batch['text'], indices):
            sents = sent_tokenize(doc)
            ids = [idx]*len(sents)
            last_sent_flags = [False]*(len(sents)-1) + [True]

            out_batch['text'] += sents
            out_batch['doc_id'] += ids
            out_batch['last_sent_flag'] += last_sent_flags

        return out_batch

    ds = ds.map(
        split_sentences,
        with_indices=True,
        batched=True,
        num_proc=cfg.params['num_proc'],
    )

    def add_indices(batch, idx):
        batch['row_id'] = idx
        return batch

    ds = ds.map(
        add_indices,
        batched=True,
        with_indices=True,
    )

    return ds


