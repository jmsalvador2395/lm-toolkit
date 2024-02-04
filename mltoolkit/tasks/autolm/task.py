
from mltoolkit.templates import Task
from .trainer import TrainerAutoLM

class TaskAutoLM(Task):

    def __init__(self, cfg, keywords, debug=False):
        super().__init__(cfg, keywords, debug)
        self.trainer_cls = TrainerAutoLM
