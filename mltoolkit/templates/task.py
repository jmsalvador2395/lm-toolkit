from mltoolkit.utils import (
    display,
)

class Task:
    
    def __init__(self, cfg, keywords, debug=False):
        self.cfg = cfg
        self.keywords = keywords
        self.debug = debug
        self.task_name = '<<Template>>'

    def train(self):
        display.error(f'Training procedure not defined for {self.task_name}')
        raise NotImplementedError()

    def evaluate(self):
        display.error(f'Evaluation procedure not defined for {self.task_name}')
        raise NotImplementedError()

    def param_search(self):
        display.error(f'Hyperparameter search procedure not defined for {self.task_name}')
        raise NotImplementedError()

    def other(self):
        display.error(f'Other procedure not defined for {self.task_name}')
        raise NotImplementedError()
