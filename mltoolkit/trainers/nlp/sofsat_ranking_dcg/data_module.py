"""
this file contains code for changing the All-Sides dataset to fit the RL Extractive Summarization task
"""

# external imports
import random
import datasets
import os
import torch
import numpy as np
from math import comb
from itertools import combinations
from typing import List, Tuple
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    AutoTokenizer,
)
from itertools import chain

# internal imports
from mltoolkit.utils import display

def fetch_dataloaders(cfg):

    ######################## set variables from cfg #######################

    batch_size = cfg.data['batch_size']
    shuffle = cfg.data.get('shuffle', True)
    num_workers = cfg.data.get('num_proc', 0)
    pin_memory = cfg.data.get('pin_memory', False)

    ######################################################################

    # first read in data as a huggingface dataset
    ds = datasets.load_dataset(
        'csv',
        data_files=cfg.data['loc'],
        cache_dir=cfg.data['cache_dir'],
        num_proc=num_workers
    )

    # TODO delete this next line
    ds = ds['train'].select(range(50))
    ds = ds.train_test_split(train_size=cfg.data['train_test_split'])

    # truncate dataset columns
    trgt_columns = {
        "left-context": "s1",
        "right-context": "s2",
        "theme-description": "intersection",
    }

    ds = ds.remove_columns(
        set(ds['train'].features.keys())
        - set(trgt_columns.keys())
    )
    for key, val in trgt_columns.items():
        ds = ds.rename_column(key, val)

    # convert to AllSides format.
    train_data, test_data = (
        AllSides(ds['train'], cfg),
        AllSides(ds['test'], cfg),
    )

    # define function for RNG in dataloaders
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(cfg.general['seed'])

    # return dataloader for train and test sets
    return (
        DataLoader(
            train_data,
            batch_size=batch_size,
            collate_fn=train_data.collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        DataLoader(
            test_data,
            batch_size=batch_size,
            collate_fn=test_data.collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        )
    )


class AllSides(Dataset):
    """
    This class wraps the huggingface-formatted all-sides dataset.

    The reason this exists is to override the __getitem__ and apply 
    a transformation on the data before being batched
    """
    
    def __init__(self, ds, cfg):

        ######################## set variables from cfg #######################

        self.name = 'all-sides'
        self.max_len = cfg.data['max_seq_len']
        self.commute_prob = cfg.data['commute_prob']
        self.n_docs = cfg.data['n_docs']
        self.dev = cfg.data['data_device']
        self.out_dev = cfg.model['device']
        self.embedder_batch_size = cfg.data['embedder_batch_size']
        self.use_cls_embs = cfg.data['use_cls_embs']
        self.truncate_docs = cfg.data['limit_sentences']
        self.sentence_limit = cfg.data['max_sentences_per_doc']

        doc_model = cfg.data['doc_model']

        ######################################################################

        # assign dataset 
        self.ds = ds

        # initialize tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = \
            AutoTokenizer.from_pretrained('distilbert-base-uncased')

        # assign embeddings model
        self.emb_model = \
            DistilBertModel.from_pretrained("distilbert-base-uncased").to(
                self.dev
            )
        self.emb_model.eval()

        self.doc_model = SentenceTransformer(
            doc_model,
            device=self.dev
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        # split sentences using sentence tokenizer
        doc_a = sent_tokenize(self.ds[idx]['s1'])
        doc_b = sent_tokenize(self.ds[idx]['s2'])
        if self.truncate_docs:
            doc_a = doc_a[:self.sentence_limit]
            doc_b = doc_b[:self.sentence_limit]

        # swap documents randomly
        swap = np.random.randint(2)
        if swap:
            doc_a, doc_b = doc_b, doc_a
        
        # get # of sentences in each document
        len_doc_a = len(doc_a)
        len_doc_b = len(doc_b)

        # make mask that separates first and second document (use 0 for [SEP])
        part_mask = torch.ones(len_doc_a + len_doc_b + 1)
        part_mask[len_doc_a] = 0
        part_mask[len_doc_a + 1:] = 2

        # concatenate documents and separate using [SEP]
        sents = doc_a + ['[SEP]'] + doc_b

        # create mask for the given sequence
        seq_mask = torch.ones(len(sents), dtype=torch.bool)
        seq_mask[len_doc_a] = 0

        sep_mask = torch.zeros(seq_mask.shape, dtype=torch.bool)
        sep_mask[len_doc_a] = 1

        return {
            'intersection': self.ds[idx]['intersection'],
            'doc_a': self.ds[idx]['s1'],
            'doc_b': self.ds[idx]['s2'],
            'part_mask': part_mask,
            'seq_mask': seq_mask,
            'sep_mask': sep_mask,
            'len_total': len_doc_a + len_doc_b + 1,
            'sents': sents,
        }

    def collate_fn(self, batch: List[Tuple]):

        # aggregate the simple things
        doc_a = [sample['doc_a'] for sample in batch]
        doc_b = [sample['doc_b'] for sample in batch]
        intersection = [sample['intersection'] for sample in batch]
        lengths = torch.tensor([sample['len_total'] for sample in batch])
        part_mask = [sample['part_mask'] for sample in batch]
        seq_mask = [sample['seq_mask'] for sample in batch]
        sep_mask = [sample['sep_mask'] for sample in batch]
        batched_sents = [sample['sents'] for sample in batch]

        # aggregate masks into padded sequences
        seq_mask = pad_sequence(
            seq_mask,
            batch_first=True,
            padding_value=0
        )

        sep_mask = pad_sequence(
            sep_mask,
            batch_first=True,
            padding_value=0
        )

        part_mask = pad_sequence(
            part_mask,
            batch_first=True,
            padding_value=0
        ).cpu().numpy()
        
        # make sentence matrix from batch
        seq_len = max(lengths)
        sent_mat = np.array([
            sent + ['']*(seq_len-len(sent))
            for sent in batched_sents
        ])

        # aggregate all individual sentences to input to the model
        sents = [sent for sample in batch for sent in sample['sents']]

        # use cls embeddings from bert model if true and embeddings from sbert model if false
        if self.use_cls_embs:
            # tokenize
            tokens = self.tokenizer(
                sents,
                max_length=self.max_len,
                padding='max_length',
                add_special_tokens=True,
                truncation=True,
                return_tensors='pt',
            ).to(self.dev)

            # collect the CLS embeddings for each sentence in all documents
            sent_embs = []
            with torch.no_grad():
                for i in range(0, len(sents), self.embedder_batch_size):
                    emb_batch = self.emb_model(
                        input_ids=tokens['input_ids'][i:i+self.embedder_batch_size],
                        attention_mask=tokens['attention_mask'][i:i+self.embedder_batch_size]
                    )
                    sent_embs.append(
                        emb_batch.last_hidden_state[:, 0, :]
                    )
            sent_embs = torch.vstack(sent_embeddings) 
        else:
            # encode individual sentences
            sent_embs = torch.tensor(
                self.doc_model.encode(sents),
                device=self.dev
            )

        # split into sample-level sequences
        indices = F.pad(lengths, (1, 0))
        indices = torch.cumsum(indices, dim=0)
        indices = zip(indices[:-1], indices[1:])
        seqs = [sent_embs[start:end] for start, end in indices]


        # convert into padded sequences for batched input
        seqs = pad_sequence(
            seqs,
            batch_first=True,
        )

        # get document embeddings
        a_embeddings = torch.tensor(
            self.doc_model.encode(doc_a),
            device=self.dev
        )

        b_embeddings = torch.tensor(
            self.doc_model.encode(doc_b),
            device=self.dev
        )

        return {
            'a': doc_a,
            'b': doc_b,
            'a_embs': a_embeddings,
            'b_embs': b_embeddings,
            'intersection': intersection,
            'part_mask': part_mask,
            'sent_mat': sent_mat,
            'seq_mask': seq_mask.to(self.out_dev),
            'sep_mask': sep_mask.to(self.out_dev),
            'sequence': seqs.to(self.out_dev),
        }
