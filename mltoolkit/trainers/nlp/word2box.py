"""
This is an implementation of the original "attention is all you need" architecture
"""
# external imports
import torch
import numpy as np
import datasets
from tqdm import tqdm
from torch import nn
from torch.nn import functional as f
from datasets import Dataset

# local imports
from mltoolkit.trainers.base import TrainerBase
from mltoolkit.models.nlp import Word2Box
from mltoolkit.utils import (
    files,
    strings,
    display,
    validate,
    tokenizers
)

class TrainerWord2Box(TrainerBase):
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug=debug)
        cfg = self.cfg
        ds_path = f'{files.project_root()}/data/wackypedia_lemma/train.csv'
        validate.path_exists(ds_path)
        self.ds = datasets.load_dataset(
            path='csv',
            data_files=ds_path,
            cache_dir=cfg.data['cache_dir'],
            num_proc=cfg.data['num_proc'],
        )

    def init_model(self):
        self.model = Word2Box(self.cfg.model)

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
        save_loc = self.save_loc

        print(strings.green(
            f'intermediate data and tools set to save to {self.save_loc}'
        ))
        tokenizer_dir = 'tokenizer/wackypedia_lemma'
        
        self.tokenizer = tokenizers.fetch_tokenizer(
            f'{save_loc}/{tokenizer_dir}',
            self.ds,
            trgt_vocab_size=cfg.model['trgt_vocab_size'],
            min_freq=cfg.model['min_freq'],
            override=self.debug,
            ver='word_level',
        )
        self.cfg.model['vocab_size'] = self.tokenizer.vocab_size


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
        tokens = self.tokenizer(
            batch['text'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(self.dev[0])

        lengths = torch.sum(
            tokens['attention_mask'],
            dim=-1,
        )
        not_empty = torch.where(lengths != 0)

        input_ids = tokens['input_ids'][not_empty].to(self.dev[0])
        attn_mask = tokens['attention_mask'][not_empty].to(self.dev[0]).to(torch.bool)
        lengths = lengths[not_empty].to(self.dev[0])
        R, C = input_ids.shape

        """
        cascade_ids = torch.vstack([
            input_ids[i][None].repeat(lengths[i]-1, 1)
            for i in range(len(lengths))
        ])
        masks = [
            f.pad(
                torch.tril(torch.ones(
                    (lengths[i]-1, lengths[i]-1),
                    dtype=torch.bool,
                    device=self.dev[0]
                )),
                pad=(0, C-lengths[i])
            )
            for i in range(len(lengths))
        ]
        cascade_masks = torch.vstack(masks)
        labels = input_ids[:, 1:][attn_mask[:, 1:]]

        """
        cascade_mask = torch.tril(attn_mask.repeat(C, 1))
        breakpoint()
        scores = self.model(cascade_ids, cascade_masks)
        print(strings.green('pass'))
        breakpoint()
        
        return loss, {
            'scalar' : {
                'loss/train' : loss,
                'accuracy/train' : accuracy
            }
        }
