

from .mnist_mlp import TaskMNIST
from .autolm import TaskAutoLM
from .neuron_skip_mlp import TaskNeuronSkipMLP
from .transformer_ae import TaskTransformerAE
from .bert import TaskBERT
from mltoolkit.utils import display

def select_task(cfg, keywords, debug):
    """
    takes in config and keywords to select and initialize the desired Task class

    Input:
      cfg: config returned from mltoolkit.cfg_reader.load()
      keywords: dictionary of keywords and values returned from mltoolkit.cfg_reader.load()

    Output: 
      Subclass of mltoolkit.Task
    """
    task_name = cfg.general['task']

    if task_name == 'mnist_mlp':
        return TaskMNIST(cfg, keywords, debug=debug)
    elif task_name == 'autolm':
        return TaskAutoLM(cfg, keywords, debug=debug)
    elif task_name == 'neuron_skip_mlp':
        return TaskNeuronSkipMLP(cfg, keywords, debug=debug)
    elif task_name == 'transformer_ae':
        return TaskTransformerAE(cfg, keywords, debug=debug)
    elif task_name == 'bert':
        return TaskBERT(cfg, keywords, debug)
    else:
        display.error(f'invalid task name: {task_name}')
        raise ValueError()
