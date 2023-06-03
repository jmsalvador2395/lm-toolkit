# external imports
import yaml
import os

# local imports
from mltoolkit.utils import (
    menus,
    files,
    validate,
    display
)
from . import trainers

def train(args):
    # open menu to find a config file if none is provided
    if args.cfg is None:
        config_path = menus.file_explorer(
            'choose a config file in',
            start_path=files.project_root() + '/cfg'
        )

    # check if the provided config path is valid
    else:
        config_path = args.cfg
        validate.path_exists(
            config_path, 
            extra_info='the given path is invalid'
        )

    # set debugging variables
    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        display.debug(f'from {__file__}: CUDA_LAUNCH_BLOCKING has been set to 1')

    # select trainer and execute training
    trainer = trainers.select(config_path, debug=args.debug)
    trainer.train()
