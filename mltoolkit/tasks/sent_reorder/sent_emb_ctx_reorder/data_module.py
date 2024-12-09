# this module is for downloading and combining the bookcorpus and wikipedia datasets and preparing them for 
# MLM and NSP

import datasets
import torch
import numpy as np
import random
from nltk import sent_tokenize
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset, concatenate_datasets
import pandas as pd
import re

from mltoolkit.utils import tensor_utils, files
from mltoolkit.models.sent_encoders.hf_models import from_hf

def load_wiki(cfg, seed=None):

    # load in wikipedia
    wiki = datasets.load_dataset(
        "wikimedia/wikipedia", 
        "20231101.en", 
        cache_dir=cfg.paths['cache'],
    )
    train_test = wiki['train'].train_test_split(test_size=2000, seed=seed)
    test_val = train_test['test'].train_test_split(
        test_size=0.5, seed=seed
    )
    wiki = DatasetDict({
        'train': train_test['train'], 'validation': test_val['train'],
        'test': test_val['test'],
    })
    def wiki_map(sample):
        sents = sent_tokenize(sample['text'])
        return {'sentences': sents}
    wiki = wiki.map(wiki_map).select_columns('sentences')

    return wiki

def load_cnndm(cfg, seed=None):

    # load in CNN/DailyMail
    cnndm = datasets.load_dataset(
        'abisee/cnn_dailymail', '3.0.0',
        cache_dir=cfg.paths['cache'],
        trust_remote_code=True,
    )
    cnndm = cnndm.rename_column('article', 'text')
    def cnndm_map(sample):
        sents = sent_tokenize(sample['text'])
        return {'sentences': sents}

    cnndm = cnndm.map(cnndm_map).select_columns('text')
    return cnndm

def load_sind(cfg, seed=None):

    sind_base = f'{files.project_root()}/data/sind'
    sind_files = {
        'train': f'{sind_base}/train.tsv',
        'validation': f'{sind_base}/dev.tsv',
        'test': f'{sind_base}/test.tsv',
    }
    sind = datasets.load_dataset(
        'text', data_files=sind_files,
    )
    def sind_xform(sample):
        sents = sample['text'].split('<eos>')
        sents = [sent.strip() for sent in sents]
        return {'sentences': sents}
    sind = sind.map(sind_xform).select_columns('sentences')

    return sind

def load_roc(cfg, seed=None):

    # load in ROCstories and randomly create splits
    base_path = f'{files.project_root()}/data/ROCStories/'
    def roc_xform(sample):
        sents = [sample[f'sentence{i}'] for i in range(1, 6)]
        return {'sentences': sents}
    roc = datasets.load_dataset(base_path, cache_dir=cfg.paths['cache'])

    train_test = roc['train'].train_test_split(test_size=0.2, seed=seed)
    test_val = train_test['test'].train_test_split(
        test_size=0.5, seed=seed
    )
    roc = DatasetDict({
        'train': train_test['train'], 'validation': test_val['train'],
        'test': test_val['test'],
    })
    roc = roc.map(roc_xform).select_columns('sentences')
    return roc

def get_dataloaders(cfg):
    """
    retrieves dataset and converts to dataloaders
    """
    seed = cfg.general['seed']

    match cfg.params.get('dataset', 'roc'):
        case 'all':
            parts = {
                'roc': load_roc(cfg, seed=seed),
                'sind': load_sind(cfg, seed=seed),
                #'cnndm': load_cnndm(cfg, seed=seed),
                #'wiki': load_wiki(cfg, seed=seed),
            }
            ag = {}
            for split in ['train', 'test', 'validation']:
                ag[split] = concatenate_datasets([
                    data[split]
                    for ds_name, data in parts.items()
                ])
            ds = DatasetDict(ag)

            # TODO delete this
            ds['test'] = parts['roc']['test']

        case 'roc':
            ds = load_roc(cfg, seed=seed)
        case 'sind':
            ds = load_sind(cfg, seed=seed)
        case 'wiki':
            ds = load_wiki(cfg, seed=seed)
        case 'cnndm':
            ds = load_cnndm(cfg, seed=seed)

    seed = cfg.general.get('seed') or random.randint(0, 2**64)
    seed_worker, g = tensor_utils.get_dl_params(seed)

    def collate_fn(batch):
        return {'sentences': [sample['sentences'] for sample in batch]}

    train_loader = DataLoader(
        ds['train'],
        batch_size=cfg.params['batch_size'],
        collate_fn=collate_fn,
        num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        ds['test'],
        batch_size=cfg.params['batch_size'],
        collate_fn=collate_fn,
        num_workers=cfg.params['num_proc'],
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader