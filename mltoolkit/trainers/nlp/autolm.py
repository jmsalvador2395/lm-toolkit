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
from transformers import PreTrainedTokenizerFast


# local imports
from mltoolkit.trainers.base import TrainerBase
from mltoolkit.models.nlp.decoder \
    import AutoregressiveTransformerDecoder \
    as AutoDecoder
from mltoolkit.utils import (
    files,
    strings,
    display,
    data,
    tokenizers,
)

class TrainerAutoLM(TrainerBase):
    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug=debug)
        cfg = self.cfg

        self.ds = datasets.load_dataset(
            path='wikitext',
            name='wikitext-103-v1',
            cache_dir=cfg.data['cache_dir'],
            num_proc=cfg.data['num_proc']
        )

    def init_model(self):
        self.model = AutoDecoder(self.cfg.model).to(self.dev[0])

    def init_optimizer(self):
        cfg = self.cfg

        # optimizer
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optim['lr'],
            betas=(
                cfg.optim.get('beta1', .9),
                cfg.optim.get('beta2', .999)
            ),
            eps=float(cfg.optim.get('eps', 1e-8)),
            weight_decay=float(self.cfg.optim.get('weight_decay', 1e-4)),
        )

    def init_loss_fn(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def prepare_data_and_tools(self):
        cfg = self.cfg
        save_loc = self.save_loc

        print(strings.green(
            f'intermediate data and tools set to save to {save_loc}'
        ))
        tokenizer_dir = \
            save_loc + \
            '/tokenizer' + \
            f'/wikitext/wikitext-2-v1'

        special_tokens = {
            'pad': '[PAD]',
            'bos': '[BOS]',
            'eos': '[EOS]',
            'unk': '[UNK]',
        }

        self.tokenizer = tokenizers.fetch_tokenizer(
            tokenizer_dir,
            self.ds,
            trgt_vocab_size=cfg.model['trgt_vocab_size'],
            min_freq=cfg.model['min_freq'],
            override=self.debug,
            ver='bpe',
            special_tokens=special_tokens,
        )
        self.cfg.model['vocab_size'] = self.tokenizer.vocab_size

    def get_model_inputs(self, batch):
        cfg = self.cfg
        seq_len = cfg.model['seq_len']
        # compute scores and calculate loss
        tokens = self.tokenizer(
            batch['text'],
            max_length=seq_len+1,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(self.dev[0])

        not_empty = torch.any(tokens['attention_mask'], dim=-1)

        # get input tokens
        input_ids = tokens['input_ids'][not_empty].to(self.dev[0])

        labels = input_ids[:, 1:]
        input_ids = input_ids[:,:-1]

        # attention mask
        attn_mask = tokens['attention_mask'][not_empty].to(self.dev[0]).to(torch.bool)
        tgt_attn_mask = attn_mask[:, 1:]
        attn_mask = \
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool)).to(self.dev[0])
        attn_mask.fill_diagonal_(False)

        return input_ids, labels, attn_mask, tgt_attn_mask

    def evaluate(self):
        cfg = self.cfg
        seq_len = cfg.model['seq_len']
        batch_size = self.cfg.data['batch_size']
        nlls = []
        for i in range(0, len(self.ds['validation']), batch_size):
            batch = self.ds['validation'][i:i+batch_size]

            input_ids, labels, attn_mask, tgt_attn_mask = \
                self.get_model_inputs(batch)

            scores = self.model(input_ids, input_ids, attn_mask, attn_mask)

            scores = scores[tgt_attn_mask]
            labels = labels[tgt_attn_mask]
            loss = self.loss_fn(scores, labels)

            nlls.append(loss)
        nll = torch.mean(torch.stack(nlls))
        ppl = torch.exp(nll)
        return ppl, {
            'scalar': {
                'loss/val' : nll,
                'ppl/val': ppl,
            },
        }

    def train_step(self, batch):

        input_ids, labels, attn_mask, tgt_attn_mask = \
            self.get_model_inputs(batch)

        scores = self.model(input_ids, input_ids, attn_mask, attn_mask)

        scores = scores[tgt_attn_mask]
        labels = labels[tgt_attn_mask]

        loss = self.loss_fn(scores, labels)
        with torch.no_grad():
            ppl = torch.exp(loss.detach())

        return loss, {
            'scalar': {
                'loss/train': loss,
                'ppl/train': ppl,
            }
        }
