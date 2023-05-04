"""
loads config files and substitutes keywords
"""
# external imports
import yaml
from collections import namedtuple

# local imports
from mltoolkit.utils import validate, files, strings
def load(path_str: str):
    """
    loads a yaml config file and substitues the keywords with pre-set values

    :param path_str: the path of the config file
    :type param: str
    """

    validate.path_exists(path_str)
    keywords = {
        'home' : files.homedir(),
        'project_root' : files.get_project_root(),
        'timestamp' : strings.now()
    }

    base_cfg = {
        'general' : None,
        'model' : None,
        'optim' : None,
        'task' : None,
        'data' : None,
    }
    categories = base_cfg.keys()

    with open(path_str, 'r') as f:
        cfg = f.read()

    cfg = strings.replace_slots(
        cfg,
        keywords
    )

    cfg = {**base_cfg, **yaml.safe_load(cfg)}
    cfg = namedtuple('Config', categories)(**cfg)
    return cfg, keywords
