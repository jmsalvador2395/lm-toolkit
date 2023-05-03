# external imports
import yaml

# local imports
from mltoolkit import utils, cfg_reader
from .trainer import Trainer

def train(args):
    # open menu to find a config file if none is provided
    if args.cfg is None:
        config_path = utils.menus.file_explorer(
            'choose a config file in',
            start_path=utils.files.get_project_root() + '/cfg'
        )

    # check if the provided config path is valid
    else:
        config_path = args.cfg
        utils.validate.path_exists(
            config_path, 
            extra_info='the given path is invalid'
        )

    # load config
    cfg = cfg_reader.load(config_path)

    trainer = Trainer(cfg)
    trainer.train()
