# external imports
import datasets
from typing import List

# local imports
from mltoolkit.utils import strings, files

def load_ds(cfg):
    """
    loads the datasets for training and testing

    :param cfg: a dictionary of settings.

    :return: two huggingface datasets
    """

    try:
        cfg['train']['path']
    except:
        msg = 'ERROR: config file has no name for the training dataset'
        print(strings.red(msg))
        raise KeyError()

    # load training_data
    ds_train = datasets.load_dataset(
        path=cfg['train']['path'],
        name=cfg['train'].get('name', None),
        data_dir=cfg['train'].get('data_dir', None),
        data_files=cfg['train'].get('data_files', None),
        sep=cfg['train'].get('sep', None),
        column_names=cfg['train'].get('column_names', None),
        split=cfg['train'].get('split', None),
        cache_dir=cfg.get('cache_dir', None),
        num_proc=cfg.get('num_proc', 1)
    )
    if isinstance(ds_train, List) :
        ds_train = datasets.concatenate_datasets(ds_train)
    elif isinstance(ds_train, datasets.DatasetDict):
        ds_train = datasets.concatenate_datasets(
            list(ds_train.values())
        )

    # load testing data
    if cfg.get('test', None) is not None:
        if cfg['test'].get('data_files', None) is not None:
            ds_test = {
                files.get_fname(data_file) : datasets.load_dataset(
                    path=pth,
                    name=None,
                    data_dir=cfg['test'].get('data_dir', None),
                    data_files=data_file,
                    sep=cfg['test'].get('sep', None),
                    column_names=cfg['test'].get('column_names', None),
                    split=cfg['test'].get('split', None),
                    cache_dir=cfg.get('cache_dir', None),
                    num_proc=cfg.get('num_proc', 1)
                )['train']
                for pth, data_file in zip(cfg['test']['path'], cfg['test']['data_files'])
            }
        else:
            ds_test = datasets.load_dataset(
                path=cfg['test']['path'],
                name=cfg['test'].get('name', None),
                data_dir=cfg['test'].get('data_dir', None),
                data_files=cfg['test'].get('data_files', None),
                sep=cfg['test'].get('sep', None),
                column_names=cfg['test'].get('column_names', None),
                split=cfg['test'].get('split', None),
                cache_dir=cfg.get('cache_dir', None),
                num_proc=cfg.get('num_proc', 1)
            )
            
        if isinstance(ds_test, List) :
            ds_test= datasets.concatenate_datasets(ds_test)
        elif isinstance(ds_test, datasets.DatasetDict):
            ds_test = datasets.concatenate_datasets(
                list(ds_test.values())
            )
    else:
        ds_test = None

    return ds_train, ds_test
