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
    trgt_dir = cfg.paths['cache'] + '/pg19_plus_wiki_docs'
    try:
        ds = datasets.load_from_disk(trgt_dir)
        ds_loaded = True
    except:
        ds_loaded = False

    if not ds_loaded:
        ds = combine_datasets(cfg)
        ds.save_to_disk(trgt_dir)
    """
    ds = datasets.load_dataset(
        'wikipedia',
        '20220301.en',
        cache_dir=cfg.paths['cache'],
        trust_remote_code=True,
    )

    ds = ds['train'].train_test_split(test_size=300)
    ds['validation'] = ds['test'].select(range(200))
    ds['test'] = ds['test'].select(range(200, 300))
    
    roc_stories = datasets.load_dataset(
        'Ximing/ROCStories',
        cache_dir=cfg.paths['cache'],
        trust_remote_code=True,
    )

    roc_stories = roc_stories.map(
        lambda x: {'text': x['prompt'] + x['continuation']},
        remove_columns=list(roc_stories['train'].features.keys()),
    )

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

    roc_loader  = DataLoader(
        roc_stories['validation'],
        batch_size=cfg.params['batch_size'],
        num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )


    return {
        'train_loader': train_loader, 
        'val_loader': val_loader, 
        'roc_stories': roc_loader,
    }

def combine_datasets(cfg):
    """
    This function performs the following:
        - loads in and then combines the pg19 and wikipedia datasets
    """

    # for extracting test samples from wikipedia
    num_test = 1000

    # load pg19 dataset
    # TODO uncomment this block
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
        #pg19['validation'],
        wiki['train'].select(range(len_wiki-num_test)),
    ])

    val_ds = datasets.concatenate_datasets([
        pg19['validation'],
        wiki['train'].select(range(len_wiki-num_test, len_wiki)),
    ])

    ds = datasets.DatasetDict({
        'train': train_ds,
        'validation': val_ds.select(range(200)),
        'test': val_ds.select(range(200, len(val_ds))),
    })
    ds = DatasetDict({
        'train': wiki['train'].select(range(len_wiki-num_test)),
        'validation': wiki['train'].select(range(len_wiki-num_test, len_wiki)),
    })
    return ds


