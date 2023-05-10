"""
this contains the function used to select the appropriate trainer based off the config model names
"""
# local imports
from mltoolkit import cfg_reader
from .example import TrainerExample
from .cv.mnist import TrainerMNIST
from .nlp.autolm import TrainerAutoLM
from .nlp.word2box import TrainerWord2Box

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
    ])
    cfg, _ = cfg_reader.load(config_path)
    match cfg.model.get('name', None):
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
        case _:
            raise ValueError(
                'invalid model name. valid options are: \n\t' +
                '\n\t'.join(models)
            )
