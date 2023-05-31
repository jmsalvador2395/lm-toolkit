"""
loads config files and substitutes keywords
"""
# external imports
import yaml
from collections import namedtuple
from random import randint

# local imports
from mltoolkit.utils import validate, files, strings
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
        'general' : {},
        'model' : {},
        'optim' : {},
        'task' : {},
        'data' : {},
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

    # set default values
    cfg = set_defaults(cfg, keywords, debug)
    cfg = set_debug(cfg, debug)

    return cfg, keywords

def set_debug(cfg, debug=False):
    if debug:
        debug_dir = f'{files.project_root()}/debug'
        cfg.general['log_dir'] = debug_dir + '/tensorboard'
        cfg.model['ckpt_dir'] = debug_dir + '/ckpt'

    return cfg

def set_defaults(cfg, keywords, debug=False):
    # set experiment name
    cfg.general['experiment_name'] = \
        f'{keywords["timestamp"]}-{cfg.model.get("name", "model")}'

    # set general parameters
    cfg.general['seed'] = cfg.general.get(
        'seed',
        randint(0, 2**32)
    )
    cfg.general['log_dir'] = cfg.general.get(
        'logdir_base',
        f'{files.project_root()}/tensorboard'
    ).rstrip('/') + \
    f'/{cfg.general["experiment_name"]}'
    cfg.general.pop('logdir_base')
    cfg.general['load_checkpoint'] = \
        cfg.model.get(
            'load_checkpoint',
            None
        )

    # set default model parameters
    cfg.model['device'] = cfg.model.get('device', 'cpu')
    cfg.model['ckpt_dir'] = (
        cfg.model.get(
            'ckpt_dir', 
            f'files.get_project_root()/checkpoints'
        ).rstrip('/') + \
        f'/{cfg.general["experiment_name"]}'
    )
    cfg.model['keep_higher_eval'] = \
        cfg.model.get(
            'keep_higher_eval',
            True
        )

    # set default optim parameters
    cfg.optim['lr'] = float(cfg.optim.get('lr', 1e-3))
    cfg.optim['weight_decay'] = float(cfg.optim.get('weight_decay', 0))
    cfg.optim['clip_max_norm'] = cfg.optim.get('clip_max_norm', None)
    cfg.optim['clip_norm_type'] = cfg.optim.get('clip_norm_type', 2.0)
    cfg.optim['swa_strat_is_linear'] = cfg.optim.get('swa_strat_is_linear', True)
    cfg.optim['swa_anneal_epochs'] = cfg.optim.get('swa_anneal_epochs', 5)
    cfg.optim['swa_lr'] = cfg.optim.get('swa_lr', 0.05)
    cfg.optim['swa_bn_update_steps'] = cfg.optim.get('swa_bn_update_steps', 0)

    # set default data parameters if they don't exist
    cfg.data['num_epochs'] = cfg.data.get('num_epochs', 1)
    cfg.data['num_shards'] = cfg.data.get('num_shards', 1)
    cfg.data['shuffle'] = cfg.data.get('shuffle', True)
    cfg.data['batch_size'] = cfg.data.get('batch_size', 32)
    cfg.data['eval_freq'] = cfg.data.get('eval_freq', 1000)
    cfg.data['log_freq'] = cfg.data.get('log_freq', 1000)
    cfg.data['using_test_loader'] = cfg.data.get('using_test_loader', False)


    return cfg
