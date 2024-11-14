# this module is for downloading and combining the bookcorpus and wikipedia datasets and preparing them for 
# MLM and NSP

import datasets
import torch
import numpy as np
import random
from nltk import sent_tokenize
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
import pandas as pd
import re

from mltoolkit.utils import tensor_utils, files
from mltoolkit.models.sent_encoders.hf_models import from_hf

def get_dataloaders(cfg):
    """
    retrieves dataset and converts to dataloaders
    """
    seed = cfg.general['seed']

    # load in SIND
    sind_base = f'{files.project_root()}/data/sind'
    sind_files = {
        'train': f'{sind_base}/train.txt',
        'validation': f'{sind_base}/dev.txt',
        'test': f'{sind_base}/test.txt',
    }
    sind = datasets.load_dataset(
        sind_base, data_files=sind_files,
        cache_dir=cfg.paths['cache'],
    )
    def sind_xform(sample):
        sents = [text.split('<eos>') for text in sample['text']]
        sents = [sent.strip() for sent in sents]
        return {'sentences': sents}
    sind = sind.with_transform(sind_xform)
    """
    sind = sind.map(
        lambda x: {
            'text': re.sub(
                '\s*[^\w\s]?\s*<eos\>\s*[^\w\s]?\s*', 
                '. ', x['text']
            )
        }
    )
    """

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
        'train': train_test['train'],
        'validation': test_val['train'],
        'test': test_val['test'],
    })
    roc = roc.with_transform(roc_xform)

    # load in CNN/DailyMail
    cnndm = datasets.load_dataset(
        'abisee/cnn_dailymail',
        '3.0.0',
        cache_dir=cfg.paths['cache'],
        trust_remote_code=True,
    )
    cnndm = cnndm.rename_column('article', 'text')
    cnndm = cnndm.select_columns('text')

    # load in wikipedia
    wiki = datasets.load_dataset(
        "wikimedia/wikipedia", 
        "20231101.en", 
        cache_dir=cfg.paths['cache'],
    )
    wiki = wiki['train'].train_test_split(test_size=100)
    wiki = wiki.select_columns('text')

    # combine data
    train_data = datasets.concatenate_datasets((
        roc['train'],
        cnndm['train'],
        wiki['train'],
    ))

    seed = cfg.general.get('seed') or random.randint(0, 2**64)
    seed_worker, g = tensor_utils.get_dl_params(seed)

    train_loader = DataLoader(
        #train_data,
        roc['train'],
        batch_size=cfg.params['batch_size'],
        num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        roc['test'],
        batch_size=cfg.params['batch_size'],
        num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader


def roc_map_fn(sample):
    sample['text'] = sample['prompt'] + ' ' + sample['continuation']
    return sample