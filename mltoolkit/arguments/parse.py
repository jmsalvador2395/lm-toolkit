"""
this is for reading in yaml files and replacing keywords with their intended values
"""
# external imports
import argparse

# internal imports
from mltoolkit.utils import (
    files,
    strings,
)

def parse():
    """
    builds and returns the program argument parser
    """
    keywords = {
        'project_root' : files.project_root(),
        'home' : files.home(),
        'timestamp' : strings.now()
    }

    # base parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--cfg',
        help='config path',
        default=None,
    )
    parser.add_argument(
        '-p',
        '--procedure',
        help='the desired procedure [\'train\', \'evaluate\', \'search\', \'other\']. (Default: train)',
        choices=['train', 'eval', 'search', 'other'],
        default='train',
    )
    parser.add_argument(
        '-d',
        '--debug',
        help='sets the program to debug mode. moves outputs to special locations (Default: False)',
        action='store_true',
        default=False,
    )

    # parse
    parser = parser.parse_args()

    if parser.cfg is not None:
        parser.cfg = strings.replace_slots(
            parser.cfg,
            keywords
        )

    return parser
