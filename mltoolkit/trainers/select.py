"""
this contains the function used to select the appropriate trainer based off the config model names
"""
# local imports
from mltoolkit import cfg_reader
from .example import TrainerExample
from .cv.mnist import TrainerMNIST
from .nlp import (
    TrainerAutoLM,
    TrainerWord2Box,
    TrainerRLExtractive,
)

def select(config_path, debug=False):
    """
    if command line tool is used to train, this section is for selecting the appropriate
    trainer to return to the train function. config is read in here and in the trainer
    to account for the use of this library as an external package
    """

    models = sorted([
        'word2box',
        'glove',
        'word2vec',
        'mnist',
        'rl_extractive',
    ])
    cfg, _ = cfg_reader.load(config_path)
    match cfg.general.get('trainer', None):
        case 'word2box':
            return TrainerWord2Box(config_path, debug=debug)
        case 'glove':
            return TrainerExample(config_path, debug=debug)
        case 'word2vec':
            return TrainerExample(config_path, debug=debug)
        case 'mnist':
            return TrainerMNIST(config_path, debug=debug)
        case 'autolm':
            return TrainerAutoLM(config_path, debug=debug)
        case 'rl_extractive':
            return TrainerRLExtractive(config_path, debug=debug)
        case _:
            raise ValueError(
                'invalid model name. valid options are: \n\t' +
                '\n\t'.join(models)
            )
