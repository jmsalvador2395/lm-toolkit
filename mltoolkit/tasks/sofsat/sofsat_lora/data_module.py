# this module loads the synthetic data generated for SOFSAT

import datasets
from torch.utils.data import DataLoader

from mltoolkit.utils import (
    tensor_utils,
    display,
)

def get_dataloaders(cfg):
    """
    retrieves dataset and converts to dataloaders
    """

    trgt_dir = cfg.paths['cache'] + '/sofsat_synthetic'
    try:
        ds = datasets.load_from_disk(trgt_dir)
        ds_loaded = True
    except Exception as e:
        display.error('place synthetic dataset in cache dir')
        import os; os._exit(1)

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
        ds['test'],
        batch_size=cfg.params['batch_size'],
        num_workers=cfg.params['num_proc'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader
