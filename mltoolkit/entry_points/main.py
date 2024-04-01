# external imports
import os

# local imports
from mltoolkit import (
    arguments, 
    unit_testing,
    utils,
    cfg_reader,
)

from mltoolkit.utils import (
    menus,
    files,
    validate,
    display
)

from mltoolkit.tasks import select_task
from mltoolkit import tasks

def main():
    """
    this is the entry point for the program
    look at the local functions imported from the top to see where the next steps are executed.
    to add more options, import the function, apply the same decorators that you see in the function
    definitions and add an 'add_argument()' entry below
    """

    # read args
    args = arguments.parse()

    # set debugging variables
    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        display.debug(f'from {__file__}: CUDA_LAUNCH_BLOCKING has been set to 1')

    # open menu to find a config file if none is provided
    if args.cfg is None:
        cfg_path = menus.file_explorer(
            'choose a config file in',
            start_path=files.project_root() + '/cfg'
        )

    # check if the provided config path is valid
    else:
        cfg_path = args.cfg
        validate.path_exists(
            cfg_path, 
            extra_info='the given path is invalid'
        )
    
    # read cfg and load task
    cfg, keywords = cfg_reader.load(cfg_path, args.debug)
    task = select_task(cfg, keywords, args.debug)

    # execute function based on given procedure
    proc = args.procedure
    if proc == 'train':
        task.train()
    elif proc == 'eval':
        task.evaluate()
    elif proc == 'search':
        task.param_search()
    elif proc == 'other':
        task.other()
