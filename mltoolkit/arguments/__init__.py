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

    # create subparser for procedures
    subparser = parser.add_subparsers(
        description='decides on which procedure to run',
        required=True,
        dest='procedure',
    )

    # add subparser for training procedure
    parser_train = subparser.add_parser('train')
    parser_train.add_argument(
        '-c',
        '--cfg',
        help='config path',
        default=None,
    )
    parser_train.add_argument(
        '-d',
        '--debug',
        help='sets the program to debug mode. moves outputs to special locations',
        action='store_true',
        default=False,
    )

    # parser for the evaluation procedure
    parser_eval = subparser.add_parser('eval')
    parser_eval.add_argument(
        '--ckpt',
        help='checkpoint path',
        default=None
    )

    # parser for unit testing
    parser_ut = subparser.add_parser('unit_test')

    parser = parser.parse_args()

    # replace with keywords
    if parser.procedure == 'train':
        if parser.cfg is not None:
            parser.cfg = strings.replace_slots(
                parser.cfg,
                keywords
            )
    elif parser.procedure == 'eval':
        if parser.ckpt is not None:
            parser.ckpt = strings.replace_slots(
                parser.ckpt,
                keywords
            )

    return parser
