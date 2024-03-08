"""
loads config files and substitutes keywords
"""
# external imports
import yaml
from collections import namedtuple
from random import randint

# local imports
from mltoolkit.utils import (
    validate,
    files,
    strings,
    display,
)
def load(path_str: str, debug=False):
    """
    loads a yaml config file and substitues the keywords with pre-set values

    :param path_str: the path of the config file
    :type param: str
    """

    validate.path_exists(path_str)
    keywords = {
        'home' : files.home(),
        'project_root' : files.project_root(),
        'timestamp' : strings.now()
    }

    base_cfg = {
        'paths': {},
        'general': {},
        'params': {},
        'search_params': {},
    }
    categories = base_cfg.keys()

    with open(path_str, 'r') as f:
        cfg = f.read()

    cfg = strings.replace_slots(
        cfg,
        keywords
    )

    # convert to named tuple
    cfg = {**base_cfg, **yaml.safe_load(cfg)}
    cfg = namedtuple('Config', categories)(**cfg)

    check_required(cfg)

    # set default values
    cfg = set_defaults(cfg, keywords, debug)

    if debug:
        debug_dir = f'{files.project_root()}/debug'

        cfg.paths['log_dir'] = \
            debug_dir \
            + f'/tensorboard' 
        display.debug(f'log dir set to {cfg.paths["log_dir"]}')


    return cfg, keywords

def set_defaults(cfg, keywords, debug=False):
    # set experiment name
    cfg.general['experiment_name'] = \
        f'{cfg.general["task"]}/{keywords["timestamp"]}'

    # set general parameters
    cfg.general['seed'] = cfg.general.get(
        'seed',
        randint(0, 2**32)
    )

    # set tensorboard logging directory
    cfg.paths['log_dir'] = cfg.general.get(
        'logdir_base',
        f'{files.project_root()}/tensorboard'
    ).rstrip('/')

    cfg.paths.pop('logdir_base')
    cfg.general['load_checkpoint'] = \
        cfg.general.get(
            'load_checkpoint',
            None
        )

    # set default optim parameters
    cfg.params['lr'] = float(cfg.params.get('lr', 1e-3))
    cfg.params['clip_max_norm'] = cfg.params.get('clip_max_norm', None)
    cfg.params['weight_decay'] = float(cfg.params.get('weight_decay', 0))
    cfg.params['clip_norm_type'] = cfg.params.get('clip_norm_type', 2.0)

    # set default data parameters if they don't exist
    cfg.params['num_epochs'] = cfg.params.get('num_epochs', 1)
    cfg.params['shuffle'] = cfg.params.get('shuffle', True)
    cfg.params['batch_size'] = cfg.params.get('batch_size', 32)
    cfg.params['eval_freq'] = cfg.params.get('eval_freq', 1000)
    cfg.params['log_freq'] = cfg.params.get('log_freq', 1000)

    return cfg

def check_required(cfg):
    # check if task is assigned
    try:
        cfg.general['task']
    except Exception as e:
        display.error('config parameter cfg.general[\'task\'] is not set')
