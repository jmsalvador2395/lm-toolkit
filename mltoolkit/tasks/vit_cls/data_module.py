# this module is for downloading and combining the bookcorpus and wikipedia datasets and preparing them for 
# MLM and NSP

import datasets
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

from mltoolkit.utils import tensor_utils

def get_dataloaders(cfg):
    """
    retrieves dataset and converts to dataloaders
    """

    ds = datasets.load_dataset('cifar100',
                                cache_dir=cfg.paths['cache'],
                                keep_in_memory=True,
                                trust_remote_code=True,).with_format('pt')

    seed_worker, g = tensor_utils.get_dl_params(cfg.general['seed'])

    train_loader = DataLoader(
        ds['train'],
        batch_size=cfg.params['batch_size'],
        num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ds['test'],
        batch_size=cfg.params['batch_size'],
        num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader
