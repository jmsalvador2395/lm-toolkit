"""
This is an implementation of the original "attention is all you need" architecture
"""
# external imports
import torch
import numpy as np
import datasets
import tokenizers
from tqdm import tqdm
from torch import nn
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers
)


# local imports
from .base import TrainerBase
from mltoolkit.utils import (
    files,
    strings,
    display,
    data
)

class TrainerAutoLM(TrainerBase):
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug=debug)
        cfg = self.cfg

        self.model = torch.nn.Linear(4,10)

        self.ds = datasets.load_dataset(
            path='wikitext',
            name='wikitext-2-v1',
            cache_dir=cfg.data['cache_dir'],
            num_proc=cfg.data['num_proc']
        )

    def init_optimizer(self):
        # optimizer
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.optim['lr'],
            weight_decay=self.cfg.optim['weight_decay']
        )

    def init_loss_fn(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def prepare_data_and_tools(self):
        cfg = self.cfg
        def batch_iterator(batch_size=cfg.data['batch_size']):
            for i in range(0, len(self.ds['train']), batch_size):
                    yield self.ds['train'][i : i + batch_size]["text"]

        if self.debug: 
            save_loc = \
                self.debug_dir + \
                '/tokenizer' + \
                f'/wikitext/wikitext-2-v1/v_{cfg.model["vocab_size"]}' + \
                f'/min_freq_{cfg.model["min_freq"]}/tokenizer.json'
        else:
            save_loc = \
                cfg.data['save_loc'] + \
                '/tokenizer' + \
                f'/wikitext/wikitext-2-v1/v_{cfg.model["vocab_size"]}' + \
                f'/min_freq_{cfg.model["min_freq"]}/tokenizer.json'
        print(strings.green(
            f'intermediate data and tools set to save to {save_loc}'
        ))

        if not files.path_exists(save_loc):
            print(strings.green(
                'tokenizer does not exist. training tokenizer ...'
            ))
            trainer = tokenizers.trainers.BpeTrainer(
                vocab_size = cfg.model['vocab_size'],
                min_frequency = cfg.model['min_freq'],
                show_progress=True,
            )
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            print(strings.green(
                'training tokenizer'
            ))
            tokenizer.train_from_iterator(
                batch_iterator(),
                trainer=trainer,
                length=len(self.ds['train']),
            )

            files.create_path(save_loc, is_file=True)
            tokenizer.save(save_loc)
            print(strings.green(
                f'tokenizer saved to {save_loc}'
            ))
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token='[UNK]',
                sep_token='[SEP]',
                pad_token='[PAD]',
                cls_token='[CLS]',
                mask_token='[MASK]',
            )

        else:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=save_loc,
                unk_token='[UNK]',
                sep_token='[SEP]',
                pad_token='[PAD]',
                cls_token='[CLS]',
                mask_token='[MASK]',
            )
            print(strings.green(
                f'loaded tokenizer from {save_loc}'
            ))

        self.tokenizer = tokenizer


    def evaluate(self):
        with torch.no_grad():
            scores = self.model(self.ds['test'][:]['image'].to(self.dev))
            labels = self.ds['test'][:]['label'].to(self.dev)
            loss = self.loss_fn(scores, labels)

        accuracy = torch.sum(torch.argmax(scores, dim=-1) == labels)/len(labels)
        return accuracy, {
            'scalar' : {
                'loss/test' : loss,
                'accuracy/test' : accuracy
            }
        }

    def train_step(self, batch):
        # compute scores and calculate loss
        inputs = self.tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        breakpoint()
        scores = self.model(batch['image'].to(self.dev))
        labels = batch['label'].to(self.dev)
        loss = self.loss_fn(scores, labels)
        accuracy = torch.sum(torch.argmax(scores, dim=-1) == labels)/len(labels)

        return loss, {
            'scalar' : {
                'loss/train' : loss,
                'accuracy/train' : accuracy
            }
        }
