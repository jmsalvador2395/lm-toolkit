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

    trgt_path = cfg.paths['cache'] + '/wiki_embeddings_small'
    roc_path = cfg.paths['cache'] + '/roc_embeddings'
    try:
        ds = datasets.load_from_disk(trgt_path)
    except:
        ds = prepare_dataset(cfg)
        ds.save_to_disk(trgt_path)

    try:
        roc_embeddings = datasets.load_from_disk(roc_path)
    except:
        roc_embeddings = prepare_roc(cfg)
        roc_embeddings.save_to_disk(roc_path)
    ds['train'] = datasets.concatenate_datasets(
        (ds['train'], roc_embeddings['train'])
    )

    ds.set_format('torch', columns=['embeddings'])
    ds = ds.filter(
        lambda x: len(x['embeddings']) >= 5,
        num_proc=cfg.params['num_proc'],
    )
    roc_embeddings.set_format('torch', columns=['embeddings'])



    seed_worker, g = tensor_utils.get_dl_params(cfg.general['seed'])

    def collate_fn(batch):
        return [x['embeddings'] for x in batch]

    train_loader = DataLoader(
        ds['train'],
        batch_size=cfg.params['batch_size'],
        #num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        ds['validation'],
        batch_size=cfg.params['batch_size'],
        #num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )

    roc_loader  = DataLoader(
        roc_embeddings['validation'],
        batch_size=cfg.params['batch_size'],
        #num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )


    return {
        'train_loader': train_loader, 
        'val_loader': val_loader, 
        'roc_stories': roc_loader,
    }

def map_fn(sample, **fn_kwargs):
    sents = sent_tokenize(sample['text'])
    embeddings = fn_kwargs['encoder'].encode(
        sents,
        convert_to_numpy=True,
        batch_size=64,
    )
    return {'embeddings': embeddings}

def prepare_roc(cfg):
    """
    This function performs the following:
        - loads in and then computes the sentence embeddings for the 
          ROCStories dataset
    """
    # sentence encoder
    model_name = 'mixedbread-ai/mxbai-embed-large-v1'
    encoder = from_hf(
        model_name, 
        emb_dim=1024, 
        max_seq_len=512,
        cache_dir=cfg.paths['cache'],
    ).to('cuda')
    
    # load data
    roc_stories = datasets.load_dataset(
        'Ximing/ROCStories',
        cache_dir=cfg.paths['cache'],
        trust_remote_code=True,
    )

    # combine text
    roc_stories = roc_stories.map(
        lambda x: {'text': x['prompt'] + x['continuation']},
        remove_columns=list(roc_stories['train'].features.keys()),
    )

    roc_embeddings = roc_stories.map(
        map_fn,
        fn_kwargs={
            'encoder': encoder,
        },
        remove_columns=list(roc_stories['train'].features.keys()),
    )

    return roc_embeddings


def prepare_dataset(cfg):
    """
    This function performs the following:
        - loads in and then computes the sentence embeddings for the 
          wikipedia dataset
    """

    # sentence encoder
    model_name = 'mixedbread-ai/mxbai-embed-large-v1'
    encoder = from_hf(
        model_name, 
        emb_dim=1024, 
        max_seq_len=512,
        cache_dir=cfg.paths['cache'],
    ).to('cuda')

    ds = datasets.load_dataset(
        'wikipedia',
        '20220301.en',
        cache_dir=cfg.paths['cache'],
        trust_remote_code=True,
    )
    ds = ds.map(
        map_fn,
        fn_kwargs={
            'encoder': encoder,
        },
        remove_columns=list(ds.features.keys()),
    )

    return ds


