"""
Extractive Summarization using RL
Model file
"""
# external imports
import gc
import re
import os
import sys
import torch
import numpy as np
import argparse
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Iterable, Union
from torch.distributions import Categorical

# local imports
from mltoolkit.models.pretrained.bert import ExtSummarizer


class MLDecoder(nn.Module):
    """
    This class taken from 'Efficient Sequence Labeling with Actor-Critic Training'
    paper code*
    """

    def __init__(
        self,
        w_rnn_units: int=768,
        dec_rnn_units: int=256,
        y_labels: int=2,
        y_em_size: int=32,
        decd_drop: float=0.5,
        *args,
        **kwargs,
    ):
        super(MLDecoder, self).__init__()
        # I am following this - https://github.com/SaeedNajafi/ac-tagger
        self.w_rnn_units = w_rnn_units
        self.dec_rnn_units = dec_rnn_units
        self.tag_em_size = y_em_size
        self.tag_size = y_labels
        self.dropout = decd_drop

        # Size of input feature vectors
        in_size = self.w_rnn_units + self.tag_em_size  # TODO: change this to y_vec
        self.dec_rnn = nn.LSTMCell(
            input_size=in_size, hidden_size=self.dec_rnn_units, bias=True
        )

        # This is a linear affine layer.
        self.affine = nn.Linear(
            self.w_rnn_units + self.dec_rnn_units, self.tag_size, bias=True
        )
        self.drop = nn.Dropout(self.dropout)
        self.param_init()
        self.embeddings()

    def param_init(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            if "weight" in name:
                nn.init.xavier_uniform_(param)
        return

    def embeddings(self):
        """Add embedding layer that maps from tag ids to tag feature vectors."""
        self.tag_em = nn.Embedding(self.tag_size, self.tag_em_size)
        # TODO: hope the following assert stateemnt won't be required
        # assert eutils.get_grad(self.tag_em) == True, 'Set it up for training'
        return

    def forward(self, sents_vec, mask_cls):
        # Note: How to compute PackedSequence (just in case)
        #  pad_inp = sents_vec * mask_cls[:, :, None]
        #  ll = mask_cls.sum(dim=1).tolist()
        #  pack_seq = nn.utils.rnn.pack_padded_sequence(pad_inp, ll, batch_first=True, enforce_sorted=False)

        H = sents_vec  # H -> Batch x Seq x Dim

        # get current batch size
        d_batch_size, max_s_len, input_dim = H.shape

        # Create a variable for initial hidden vector of RNN.
        h0 = torch.zeros(d_batch_size, self.dec_rnn_units).type_as(H)

        # Create a variable for the initial previous tag.
        # This is like a PAD tag. You start from zero
        Go_symbol = torch.zeros(d_batch_size, self.tag_em_size).type_as(H)

        Scores = []
        # use these for first token
        prev_y_em = Go_symbol
        prev_h = h0
        prev_c = h0
        for i in range(max_s_len):  # iterate over tokens in sequence
            Hi = H[:, i, :]  # get vector for ith token
            mask_idx = mask_cls[:, i, None]
            input_vec = torch.cat((prev_y_em, Hi), dim=1)

            # Note: output is h and c is c (ref: image in prev link)
            #  (https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2)
            h_idx, c_idx = self.dec_rnn(input_vec, (prev_h, prev_c))

            # TODO:
            #   Loss-Computation -> Batch sample j --> [y1, y2, y3, ..., yi, yp, ..., yp] (*) mask_cls[j, :]
            # process the output (h) to convert into y
            output_dr = self.drop(h_idx)  # [batch, 256]  #  applying dropout on output
            output_dr_H = torch.cat((output_dr, Hi), dim=1)  # [batch, 1024]
            score = self.affine(output_dr_H)  # for each word get logits

            pred_y = torch.argmax(
                score, dim=-1
            )  # sample the value (based on Reinforce algo)
            pred_y_em = self.tag_em(pred_y)
            Scores.append(score)

            # Note: Following FB code for handling pad vectors.
            #  (https://github.com/facebookresearch/latent-treelstm/blob/master/modules/LstmRnn.py)
            #  Works as per my maths
            #  I believe it does not affect whether I do it or not as long as
            #  I multiply the predcited probability (one number) with mask
            # For the next step
            prev_h = h_idx * mask_idx + prev_h * (1.0 - mask_idx)
            prev_c = c_idx * mask_idx + prev_c * (1.0 - mask_idx)

            # Teachor Force the previous gold tag.
            prev_y_em = pred_y_em

        # Return log_probs
        return nn.functional.log_softmax(torch.stack(Scores, dim=1), dim=2)

class RLModel(nn.Module):
    """Main RL model"""

    def __init__(self, cfg):
        super().__init__()

        # assign values from cfg and set defaults otherwise
        base_model = cfg.get('base_model', 'distilbert')
        n_docs = cfg.get('n_docs', 2)
        permute_prob = cfg.get('permute_prob', 0)
        freeze_base = cfg.get('freeze_base', True)
        ckpt_dir = cfg.get('base_model_ckpt', None)

        # decoder parameters
        w_rnn_units = cfg.get('w_rnn_units', 768)
        dec_rnn_units = cfg.get('dec_rnn_units', 256)
        y_em_size = cfg.get('y_em_size', 256)
        decd_drop = cfg.get('decd_drop', .1)
        output_features = cfg.get('output_features', 2)
        self.k = cfg.get('budget', 3)
        self.stay_in_budget = cfg.get('stay_in_budget', False)

        # check check parameter values
        assert 1.0 >= permute_prob >= 0.0, "Prob is out of range [0, 1]"
        if ckpt_dir is None:
            raise ValueError('checkpoint path, <ckpt_dir>, is required')

        self.permute_prob = permute_prob
        self.n_docs = n_docs

        orig_model = ExtSummarizer(
            checkpoint=torch.load(
                ckpt_dir, map_location="cpu"
            ),
            bert_type=base_model,
            device="cpu",
        )

        self.base_model = orig_model.bert
        self.ext_layer = orig_model.ext_layer
        self.decoder = MLDecoder(
            w_rnn_units=w_rnn_units,
            dec_rnn_units=dec_rnn_units,
            y_labels=output_features,
            y_em_size=y_em_size,
            decd_drop=decd_drop,
        )

        # # OLD
        if freeze_base:
            self.freeze_params(self.base_model)
            self.assert_all_frozen(self.base_model)

            # Added extra to see the performance
            self.freeze_params(self.ext_layer)
            self.assert_all_frozen(self.ext_layer)

        # self._trash([checkpoint, ])

    @staticmethod
    def freeze_params(model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    @staticmethod
    def assert_all_frozen(model):
        """Assert that parameters are frozen"""

        def grad_status(model: nn.Module) -> Iterable:
            return (par.requires_grad for par in model.parameters())

        def lmap(f: Callable, x: Iterable) -> List:
            """list(map(f, x))"""
            return list(map(f, x))

        model_grads: List[bool] = list(grad_status(model))
        n_require_grad = sum(lmap(int, model_grads))
        npars = len(model_grads)
        assert not any(
            model_grads
        ), f"{n_require_grad / npars:.1%} of {npars} weights require grad"

    @staticmethod
    def _trash(vars: list):
        """Utility function to manually clear the garbage"""
        for v in vars:
            del v
        gc.collect()

    def forward(self, input_ids, **kwargs) -> Tuple:
        sents_vec_list = []
        mask_cls_list = []

        for doc in input_ids.keys():
            doc_data = argparse.Namespace(**input_ids[doc])
            top_vec = self.base_model(
                doc_data.src, doc_data.segs, doc_data.mask_src
            )  # [Batch, #token, Hidden_SZ]
            sents_vec = top_vec[
                torch.arange(top_vec.size(0)).unsqueeze(1), doc_data.clss
            ]
            sents_vec_list.append(sents_vec * doc_data.mask_cls[:, :, None].float())
            mask_cls_list.append(doc_data.mask_cls)

        # concatenate across token dimension
        sents_vec = torch.cat(sents_vec_list, dim=1)
        mask_cls = torch.cat(mask_cls_list, dim=1)

        """
        After concatenating move all the zeros in mask to the side
        Useful for decoder for computing  
        Reasons to do it :
                1. Save a lot of computation (https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch#:~:text=And%20we%20do%20packing%20so,would%20affect%20the%20overall%20performance.)
                2. I won't have to add pad token to tag_embeds (in decoder)
                   if not done, for input pad, we would predict 1 or 0
                   whereas we explicitly want this to be pad.
                   
        1. PackedSequence not useful to use since we are using LSTM cell 
        """
        N, S = mask_cls.shape
        x_grid, y_grid = np.indices((N, S))  # x, y indices of the grid

        # Move all zero/pads to the end  # working
        mask_cls, m_idxs = move_zeros_to_end(mask_cls)
        sents_vec = sents_vec[x_grid, m_idxs, :].type_as(sents_vec)  # New and better
        # sents_vec = sents_vec[torch.arange(N).unsqueeze(1), m_idxs, :].type_as(sents_vec)      # Old

        # Remove the cols which are all zeros (from the end since zeros are at the end after previous step)
        # Note: use trim_zeros function (not available in torch)
        non_empty_mask = self._get_non_empty_mask(mask_cls)
        sents_vec = sents_vec[:, non_empty_mask, :]
        mask_cls = mask_cls[:, non_empty_mask]
        sents_vec = self.ext_layer(sents_vec, mask_cls, my_flag=True)

        # After trimming y dim might change
        _, S = mask_cls.shape
        x_grid, y_grid = np.indices((N, S))

        # permute prob - prob of shuffling the sentences (bag of sentences)
        permute = np.random.choice(
            [False, True], p=[1 - self.permute_prob, self.permute_prob]
        )
        shuffled_grid = None
        if sys.gettrace() is not None:  # Debug Mode
            permute = True
        if permute:
            sent_counts = mask_cls.abs().sum(dim=1).tolist()
            shuffled_grid = self._get_shuffled_idxs(sent_counts, S)
            sents_vec = sents_vec[x_grid, shuffled_grid, :].type_as(sents_vec)

        log_softmax_scores = self.decoder(
            sents_vec, mask_cls
        )  # log softmax scores, mask_cls, y_grid

        # Get back the original order
        assert not torch.any(
            torch.isnan(log_softmax_scores)
        ), "Nans should not occur. Before repermuting"
        # log_softmax_scores_orig = log_softmax_scores.clone()  # clone them to rectify Nan error
        rev_argsort = None
        if permute:
            # Note: Idea is double argsort but in efficinet way (Source - https://stackoverflow.com/a/28574015)
            rev_argsort = np.empty((N, S), dtype=np.intp)
            rev_argsort[x_grid, shuffled_grid] = y_grid
            log_softmax_orig_order = log_softmax_scores[x_grid, rev_argsort, :].type_as(
                sents_vec
            )
            cond = torch.all(
                log_softmax_orig_order[x_grid, shuffled_grid] == log_softmax_scores
            )
            assert cond, "Sanity check that I am getting the original order"
            log_softmax_scores = log_softmax_orig_order

        assert not torch.any(
            torch.isnan(log_softmax_scores)
        ), "Nans should not occur. After repermuting"


        # sample actions
        preds = \
            Categorical(torch.exp(log_softmax_scores)).sample()

        if self.stay_in_budget:
            preds, firstk_mask, episode_lengths = get_firstk(
                preds,
                self.k,
                shuffled_grid,
                rev_argsort
            )
            mask_cls = firstk_mask
        else:
            mask_cls = mask_cls.bool()
            episode_lengths = torch.sum(mask_cls, dim=-1)

        return log_softmax_scores, preds, episode_lengths, mask_cls


    @staticmethod
    def _get_shuffled_idxs(sent_cts: list, max_sent_ct: int) -> np.ndarray:
        """
        Bag of sentences model/ Data augmentation
        Shuffles the indices of sentences (except for pad tokens so that they are always at the end)
        """

        def do_shuffled_idxs(count: int):
            return np.concatenate(
                (random_idxs(count), np.arange(count, max_sent_ct))
            )

        return np.array([do_shuffled_idxs(ct) for ct in sent_cts])

    @staticmethod
    def _get_non_empty_mask(mask_cls: torch.Tensor) -> torch.Tensor:  # working
        """
        Mask to say whether a col has all zeros (0 for all zero values)
        Note: You can use this apporach too -
        mask = np.all(A == 0, axis=0)
        data[:, ~mask]
        """
        non_empty_mask: torch.Tensor = mask_cls.abs().sum(dim=0)
        # asset that all zeros are at end
        zero_loc = torch.where(non_empty_mask == 0)[0]
        non_zero_loc = torch.where(non_empty_mask != 0)[0]
        cond = torch.all(
            torch.cat([non_zero_loc, zero_loc])
            == torch.arange(non_empty_mask.nelement()).type_as(mask_cls)
        )
        assert cond, "Mistake in non-empty mask creation"
        return non_empty_mask.bool()

def get_firstk(preds, k=3, shuffled_grid=None, unshuffled_grid=None):


    # set vars
    dev = preds.device
    N, S= preds.shape

    # initialize mask array
    firstk_mask = torch.full((N, S), True, device=dev)

    # intialize array that keeps count of 1s for each episode in the batch
    counts = torch.zeros(N, device=dev)

    shuffled = True if shuffled_grid is not None else False
    if shuffled:
        x_grid, _ = np.indices((N, S))
        shuffled_preds = preds[x_grid, shuffled_grid]

    # iterate through the columns
    for i in range(S):
        # set element to False if k 1s have already been encountered
        firstk_mask[:, i] = counts < k
        if shuffled:
            counts[shuffled_preds[:, i] == 1] += 1
        else:
            counts[preds[:, i] == 1] += 1

    if shuffled:
        firstk_mask = firstk_mask[x_grid, unshuffled_grid]

    
    # set all elements after first k to 0
    preds = preds * firstk_mask

    # compute episode lengths
    episode_lengths = torch.sum(firstk_mask, dim=-1)
    episode_lengths[episode_lengths == 0] = S

    return preds, firstk_mask, episode_lengths



def move_zeros_to_end(mask_cls: torch.Tensor) -> tuple:  # perfect
    """Moves all the zeros/pad tokens to end in each row"""
    # Note: this should be slow but surprisingly fast compared to my sol
    moved_mask, moved_idxs = mask_cls.sort(dim=1, stable=True, descending=True)

    return moved_mask, moved_idxs

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
