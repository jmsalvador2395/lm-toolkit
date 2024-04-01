
from mltoolkit.templates import Task
from .trainer import TrainerSentEmbedReordering

class TaskSentEmbedReordering(Task):

    def __init__(self, cfg, keywords, debug=False):
        super().__init__(cfg, keywords, debug)
        self.trainer_cls = TrainerSentEmbedReordering
