"""
this file contains code for changing the All-Sides dataset to fit the RL Extractive Summarization task
"""

# external imports
import string
import random
import datasets
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
import torch
import math
import numpy as np
import itertools

def fetch_dataloaders(cfg, tokenizer):

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
    ds = ds['train'].train_test_split(
        test_size=1-cfg.data['train_test_split'],
        train_size=cfg.data['train_test_split'],
    )
    ds = set_columns(ds)

    # convert to AllSides format.
    train_data, test_data = (
        AllSides(ds['train'], cfg.data, tokenizer),
        AllSides(ds['test'], cfg.data, tokenizer),
    )

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
    
    def __init__(self, ds, cfg, tokenizer):

        ######################## set variables from cfg #######################

        self.name = 'all-sides'
        self.max_len = cfg['max_seq_len']
        self.commute_prob = cfg['commute_prob']
        self.n_docs = cfg['n_docs']

        ######################################################################

        # assign dataset and tokenizer
        self.ds = ds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        # insert '[CLS]' and '[SEP]' tokens
        s1, s1_len = preprocess(
            self.ds[idx]['s1'],
            self.tokenizer,
            self.max_len,
        )
        s2, s2_len = preprocess(
            self.ds[idx]['s2'],
            self.tokenizer,
            self.max_len
        )
        
        # pass to tokenizer
        s1_tokenized = load_tokenized_text(
            s1,
            s1_len,
            self.max_len,
            self.tokenizer
        )
        s2_tokenized = load_tokenized_text(
            s2,
            s2_len,
            self.max_len,
            self.tokenizer
        )

        return (
            (s1_tokenized, s2_tokenized),
            self.ds[idx]['intersection']
        )

    def collate_fn(self, batch: List[Tuple]):
        """Properly handles batching of data. Imp from training"""
        out = dict()
        out["tgt_txt"] = [sample[1] for sample in batch]

        # Apply commutative property
        doc_idxs = np.arange(self.n_docs)
        if np.random.choice([0, 1], p=[1 - self.commute_prob, self.commute_prob]):
            doc_idxs = random_idxs(self.n_docs)

        for doc in doc_idxs:
            doc_batch = [sample[0][doc] for sample in batch]
            out[f"s{doc + 1}"] = self._collate(doc_batch)

        return out

    def _collate(self, doc_batch: List[Tuple]):  # perfect
        pad_id = self.tokenizer.vocab["[PAD]"]
        pre_src = [x[0] for x in doc_batch]
        pre_segs = [x[1] for x in doc_batch]
        pre_clss = [x[2] for x in doc_batch]
        src_sents = [x[3] for x in doc_batch]

        src = torch.tensor(self._pad(pre_src, pad_id))  ## 0 is pad token
        segs = torch.tensor(self._pad(pre_segs, pad_id))

        # Naman: Both statements are same
        # mask_src = 1 - (src == 0)*torch.ones_like(src)
        mask_src = (src != 0) * torch.ones_like(src)

        # Naman:
        """
		Reason he's padding with -1 is because first CLS token will be at 0 position. 
		Thus, would create conflict while creating mask (mask_cls). 
		Later on, he explicitly changes it back to 0.
		"""
        clss = torch.tensor(self._pad(pre_clss, -1))
        mask_cls = (clss != -1) * torch.ones_like(clss)
        clss[clss == -1] = 0  # clss pad (-1) has been explicitly set to 0 here

        # TODO: check where 'src_sent_labels' is used in train code

        return {
            "src": src,
            "mask_src": mask_src,
            "segs": segs,
            "clss": clss,
            "mask_cls": mask_cls,
            "src_sents": src_sents,
        }

    def _pad(self, data, pad_id: int, width: int = -1):
        """Utility function which pads till the max sentence length in batch. Save training time"""
        if width == -1:
            width = max(len(d) for d in data)

        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data




def set_columns(ds):
    """
    renames dataset columns and removes those that aren't mentioned in the <trgt_columns> variable

    Input 
        ds[datasets.Dataset]: the all-sides dataset in huggingface format

    Output
        out_ds[datasets.Dataset]: the all-sides dataset with renamed columns only
    """

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

    return ds

def preprocess(source_text: str, tokenizer, max_len: int=512, debug=False):
    """
    text preprocessor that inserts '[CLS]' and '[SEP]' tokens between each sentence
    """
    
    cls_sep = ' [CLS] [SEP] '
    raw_text = source_text.replace("\n", " ").replace(cls_sep, " ")
    sents = sent_tokenize(raw_text)

    temp = []
    for sent in sents:
        sent = sent.strip()
        sent_wo_punctuation = (
            sent.translate(str.maketrans("", "", string.punctuation))
        ).strip()
        if sent_wo_punctuation:
            temp.append(sent)

    sents = temp
    # Naman: Modified original
    # 	https://github.com/chriskhanhtran/bert-extractive-summarization/issues/2

    ##################################################################
    # Naman: the next part is to randomize the sentences in a document.
    #  But it is slightly complicated to make sure same process is followed as baseline methods
    raw = " ".join(sents)
    raw = raw.strip().lower()
    src_subtokens = tokenizer.tokenize(raw, truncation=True)
    src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
    src_subtoken_idxs = src_subtoken_idxs[: max_len - 2]
    truncated_text = tokenizer.decode(
        src_subtoken_idxs, clean_up_tokenization_spaces=True
    )

    truncated_text = truncated_text.strip()
    if truncated_text[-1] not in '.?!"': 
        truncated_text += '.'
    sents = sent_tokenize(truncated_text)

    # TODO might have to delete this
    #random.Random().shuffle(sents)

    processed_text = cls_sep.join(sents)  # mine

    return processed_text, len(sents)

def load_tokenized_text(processed_text: str, n_sents: int, max_len, tokenizer) -> tuple:  # perfect
    """Tokenizes the text"""
    max_pos = max_len
    sep_vid = tokenizer.vocab["[SEP]"]  # 102
    cls_vid = tokenizer.vocab["[CLS]"]  # 101
    pad_vid = tokenizer.vocab["[PAD]"]  # 0
    unk_vid = tokenizer.vocab["[UNK]"]  # 100

    def _process_src(raw: str) -> tuple:
        raw = raw.strip().lower()
        raw = raw.replace("[cls]", "[CLS]").replace(
            "[sep]", "[SEP]"
        )  # since he had lowered it
        src_subtokens = tokenizer.tokenize(raw, truncation=True)
        src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(
            src_subtokens
        )  # words to ids

        # truncate to max length
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid

        # get back the original text but after truncation
        truncated_text = tokenizer.decode(
            src_subtoken_idxs, clean_up_tokenization_spaces=True
        )

        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        # ####################################
        # Naman:
        """
        # Original:
        # src: torch.Tensor = torch.tensor(src_subtoken_idxs)[None, :]	# this is perfect
        # mask_src: torch.Tensor = (1 - (src == 0).float())  # whenever padding is make mask = 0
        # cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        # clss: torch.Tensor = torch.tensor(cls_ids)  # tensor
        # mask_cls: torch.Tensor = 1 - (clss == -1).float()  # this is because clss is padded by -1 but no effect here
        # clss[clss == -1] = 0	# tensor
        # return src, mask_src, segments_ids, clss, mask_cls
        """

        # Mine
        src = src_subtoken_idxs
        clss = [i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]
        clss = [0 if ii == -1 else ii for ii in clss]
        return src, segments_ids, clss, truncated_text

    # ####################################
    # Naman:
    # Original:
    # src, mask_src, segments_ids, clss, mask_cls = _process_src(processed_text)
    # segs = torch.tensor(segments_ids)[None, :]
    # src_text = [[sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")]]  # list of list of sentences
    # return src, mask_src, segs, clss, mask_cls, src_text, n_sents
    # Mine
    src, segs, clss, truncated_text = _process_src(processed_text)
    orig_text = [
        sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")
    ]  # list of sentences
    truncated_text = [
        sent.replace("[SEP]", "").strip()
        for sent in truncated_text.split("[CLS]")
        if not sent.strip() == ""
    ]  # list of sentences
    assert len(orig_text) == n_sents, "Number of sentences should match"
    assert len(truncated_text) == len(clss), (
        f"Number of sentences in truncated text "
        f"should be equal to number of CLS tokens"
    )
    return src, segs, clss, truncated_text

def random_idxs(count: int) -> np.ndarray:
    """Returns random indices"""
    if (
        count <= 3
    ):  # in smaller sequence permutation does not necessarily give different order
        tot_permutations = math.factorial(count)
        all_perm_gen = itertools.permutations(range(count))
        # low (inclusive) to high (exclusive), 0 index (orig oder) not included
        random_perm_idx = np.random.randint(low=1, high=tot_permutations)
        sfle = np.asarray(
            next(itertools.islice(all_perm_gen, random_perm_idx, None))
        )  # tuple -> array
    else:
        sfle = np.random.permutation(count)
        while all(sfle == np.arange(count)):  # new and old sequence are equal
            sfle = np.random.permutation(count)
    return sfle
