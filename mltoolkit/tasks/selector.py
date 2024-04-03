

from .mnist_mlp import TaskMNIST
from .autolm import TaskAutoLM
from .neuron_skip_mlp import TaskNeuronSkipMLP
from .transformer_ae import TaskTransformerAE
from .bert import TaskBERT
from .vit_cls import TaskVitCls
from .sofsat import TaskSofsatLora
from .sent_reorder import (
    TaskSentEmbedReordering,
    TaskSentEmbedReordering5,
)
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

    task_dict = {
        'mnist_mlp': TaskMNIST,
        'autolm': TaskAutoLM,
        'neuron_skip_mlp': TaskNeuronSkipMLP,
        'transformer_ae': TaskTransformerAE,
        'bert': TaskBERT,
        'vit_cls': TaskVitCls,
        'sofsat/lora': TaskSofsatLora,
        'sent_emb_reorder': TaskSentEmbedReordering,
        'sent_emb_reorder_5': TaskSentEmbedReordering5,
    }
    task = task_dict.get(task_name, None)

    if task is not None:
        return task(cfg, keywords, debug=debug)
    else:
        display.error(f'invalid task name: {task_name}')
        raise ValueError()