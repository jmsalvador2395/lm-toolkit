"""
this contains the function used to select the appropriate trainer based off the config model names
"""
# local imports
from .example import TrainerExample

def select(cfg, keywords):

    match cfg.model.get('name', None):
        case 'word2box':
            return TrainerExample(cfg, keywords)
        case 'glove':
            return TrainerExample(cfg, keywords)
        case 'word2vec':
            return TrainerExample(cfg, keywords)
            
