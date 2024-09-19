# this module is for downloading and combining the bookcorpus and wikipedia datasets and preparing them for 
# MLM and NSP

import datasets
import torch
import numpy as np
import random
from nltk import sent_tokenize
from torch.utils.data import DataLoader
from datasets import DatasetDict

from mltoolkit.utils import tensor_utils
from mltoolkit.models.sent_encoders.hf_models import from_hf

def get_dataloaders(cfg):
    """
    retrieves dataset and converts to dataloaders
    """

    """
    # set the save directory of the dataset to be created
    trgt_dir = cfg.paths['cache'] + 'sent_emb_reorder'

    # try to load the dataset
    try:
        ds = datasets.load_from_disk(trgt_dir)
        ds_loaded = True
    except:
        ds_loaded = False
    # loads the dataset from scratch if no local copy is found
    if not ds_loaded:
    """

    # TAB HERE
    # load in ROCstories
    roc = datasets.load_dataset(
        'Ximing/ROCStories',
        cache_dir=cfg.paths['cache'],
        trust_remote_code=True,
    )
    roc = roc.map(roc_map_fn)
    roc = roc.select_columns('text')
    #roc.rename_column

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

    ###### TAB HERE

    seed_worker, g = tensor_utils.get_dl_params(cfg.general['seed'])

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