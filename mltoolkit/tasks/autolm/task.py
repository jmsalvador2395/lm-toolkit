
from mltoolkit.templates import Task
from .trainer import TrainerAutoLM

class TaskAutoLM(Task):

    def __init__(self, cfg, keywords, debug=False):
        super().__init__(cfg, keywords, debug)
        self.trainer_cls = TrainerAutoLM

    def train(self):
        trainer = self.trainer_cls(self.cfg, self.debug)
        trainer.train()
    """

    def evaluate(self):
        display.error(f'Evaluation procedure not defined for {self.task_name}')
        raise NotImplementedError()

    def param_search(self):
        display.error(f'Hyperparameter search procedure not defined for {self.task_name}')
        raise NotImplementedError()

    def other(self):
        display.error(f'Other procedure not defined for {self.task_name}')
        raise NotImplementedError()
    """
