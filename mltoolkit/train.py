# external imports
import yaml

# local imports
from mltoolkit import utils
from . import trainers

def train(args):
    # open menu to find a config file if none is provided
    if args.cfg is None:
        config_path = utils.menus.file_explorer(
            'choose a config file in',
            start_path=utils.files.project_root() + '/cfg'
        )

    # check if the provided config path is valid
    else:
        config_path = args.cfg
        utils.validate.path_exists(
            config_path, 
            extra_info='the given path is invalid'
        )

    trainer = trainers.select(config_path, debug=args.debug)
    trainer.train()
