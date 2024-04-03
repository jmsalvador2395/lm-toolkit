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

    trgt_dir = cfg.paths['cache'] + '/ROCStories-mxbai'
    try:
        ds = datasets.load_from_disk(trgt_dir)
        ds_loaded = True
    except:
        ds_loaded = False
    if not ds_loaded:
        ds = datasets.load_dataset(
            'Ximing/ROCStories',
            cache_dir=cfg.paths['cache'],
            trust_remote_code=True,
        )
        model_name = 'mixedbread-ai/mxbai-embed-large-v1'
        encoder = from_hf(
            model_name, 
            emb_dim=1024, 
            max_seq_len=512,
            cache_dir=cfg.paths['cache'],
        )

        ds = ds.map(
            mapper_fn, 
            fn_kwargs={'encoder': encoder},
        )
        del(encoder)
        torch.cuda.empty_cache()
        ds = ds.filter(lambda x: len(x['embeddings']) == 5)
        ds.save_to_disk(trgt_dir)
    ds = ds.with_format(
        'torch', 
        columns=['embeddings'],
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

    return train_loader, val_loader

def mapper_fn(sample, **fn_kwargs):
    """
    Pre-computes the sentence embeddings for the ROCStories data
    """
    sents = [sample['prompt']] + sent_tokenize(sample['continuation'])
    embs = fn_kwargs['encoder'].encode(sents)
    sample['embeddings'] = embs
    return sample