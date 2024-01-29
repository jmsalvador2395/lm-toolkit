"""
this file contains code for the model used to compute rewards for the rl_extractive trainer
"""
# external imports
import re
import torch
import numpy as np
import pandas as pd
import itertools
from torch import nn
from torch.nn import functional as F
from sentence_transformers import (
    SentenceTransformer,
    util,
)
from typing import (
    Tuple,
    List,
    Dict,
)

# internal imports
from mltoolkit.utils import display


class RewardModel(nn.Module):
    
    def __init__(self, cfg):
        super(RewardModel, self).__init__()

        ######################## set variables from cfg #######################

        self.model = SentenceTransformer(
            cfg.model.get(
                'reward_model',
                'all-mpnet-base-v1'
            )
        )
        self.batch_size = cfg.model['reward_batch_size']
        self.procedure = cfg.general['trainer']
        self.out_dev = cfg.model['device']

        ######################################################################

    def forward(self, batch, preds):
        if self.procedure == 'sofsat-extractive-rl':
            return evaluate_episodes(batch, preds)
        elif self.procedure == 'sofsat_extractive_sup':
            return get_labels(batch, preds)

    def get_scores(self, batch: Dict):
        
        dev = self.model.device

        ########################## unpack batch dictionary ########################## 

        intersection = batch['intersection']
        part_mask = batch['part_mask']
        seq_mask = batch['seq_mask'].cpu().numpy()
        sent_mat = batch['sent_mat']

        ############################################################################# 

        N, S = sent_mat.shape
        sent_embeddings = self.model.encode(
            sent_mat.reshape((-1,)),
            batch_size=self.batch_size,
            device=dev,
            convert_to_tensor=True,
        )

        # build set masks from predictions
        masks = {
            'sentence': sentence_embeddings,
            'a-b': flipped_preds * (batch['part_mask'] == 1),
            'b-a': flipped_preds * (batch['part_mask'] == 2),
        }

        breakpoint()
        
    def build_mask(self, preds, seq_mask, k):

        ############ set variables ############ 

        dev = self.model.device
        N, S = preds.shape
        
        # tensors
        zero = torch.tensor(0, device=dev)
        neg_one = torch.tensor(-1, device=dev)
        preds = preds.to(dev)
        seq_mask = seq_mask.to(dev)
        out_idx = torch.full((N,), -1, dtype=torch.int32, device=dev)
        out_ids = -torch.ones(N, k, device=dev)

        #######################################


        for i in range(S):
            
            step = preds[:, i]
            in_seq = seq_mask[:, i]

            #zeros = (step == 0)
            ones = (step == 1)
            twos = (step == 2)
            threes = (step == 3)

            # if a=1, store sentence index to out_ids and increment out_idx
            mask = ones & (out_idx < (k-1)) & in_seq
            out_ids[mask, out_idx[mask]] = i
            out_idx[mask] += 1

            # if a=2, store sentence index to out_idx-1
            # ReLU is used to make sure index does't go negative
            mask = twos & in_seq
            out_ids[mask, torch.max(out_idx[mask]-1, neg_one)] = i

            # if a=3, store sentence index to out_idx-2 and decrement out_idx.
            # ReLU is used to make sure index does't go negative
            mask = threes & in_seq
            out_ids[mask, torch.max(out_idx[mask]-2, neg_one)] = i
            out_idx[mask] = torch.max(out_idx[mask] - 1, neg_one)

        breakpoint()

    def evaluate_episodes(
        self,
        batch: Dict,
        preds: torch.Tensor,
        action_space: int=2,
        k: int=3,
        ver: int=1,
    ):
        if ver == 0:
            return self._ver0(batch, preds, action_space, k)
        elif ver == 1:
            return self._ver1(batch, preds, action_space, k)

    def _ver0(
        self,
        batch: Dict,
        preds: torch.Tensor,
        action_space: int=2,
        k: int=3,
    ):


        dev = self.model.device

        ########################## unpack batch dictionary ########################## 

        intersection = batch['intersection']
        part_mask = batch['part_mask']
        seq_mask = batch['seq_mask']
        sent_mat = batch['sent_mat']

        ############################################################################# 
        
        if action_space == 2:
            # mask out irrelevant indices and convert to numpy
            sent_mask = seq_mask.cpu().numpy() * preds.bool().cpu().numpy()
            flipped_mask = ~sent_mask

        elif action_space == 4:
            sent_mask = self.build_mask(preds, seq_mask, k)
            breakpoint()

        # build set masks from predictions
        masks = {
            'intersect(a, b)': sent_mask,
            'a-b': flipped_mask * (batch['part_mask'] == 1),
            'b-a': flipped_mask * (batch['part_mask'] == 2),
        }
        
        # build documents by indexing sent_mat using the masks
        docs = {key: [] for key in masks.keys()}
        make_doc = lambda doc, mask: ' '.join(doc[mask])
        for i, doc in enumerate(sent_mat):
            for key, mask in masks.items():
                docs[key].append(make_doc(doc, mask[i]))

        # compute embeddings for each document set
        embeddings = {
            key: self.model.encode(
                doc_set,
                batch_size=self.batch_size,
                device=dev,
                convert_to_tensor=True,
            )
            for key, doc_set in docs.items()
        }
        # append embeddings from documents a and b 
        embeddings['a'] = batch['a_embs']
        embeddings['b'] = batch['b_embs']

        scores = self.sim_scores(
            embeddings,
            'intersect(a, b)',
            mode='doc'
        )
        
        # compute reward
        G = (
            scores['a, intersect(a, b)']
            + scores['b, intersect(a, b)']
            - scores['a, b-a']
            - scores['b, a-b']
            - scores['a-b, intersect(a, b)']
            - scores['b-a, intersect(a, b)']
            - scores['a-b, b-a']
        )

        # convert similarity keys for metric logging
        metrics= {
            f'reward/sim({key})': val.cpu().numpy() 
            for key, val in scores.items()
        }

        # set negative terms for metric visuals
        pos_terms = [
            'reward/sim(a, intersect(a, b))',
            'reward/sim(b, intersect(a, b))'
        ]
        for key in metrics:
            if key not in pos_terms:
                metrics[key] = -metrics[key]
        metrics['reward/aggregate'] = G.cpu().numpy()

        return (
            G.to(self.out_dev), # return reward for loss function
            metrics, # return sim scores for metric logging
            docs['intersect(a, b)'], # return document predictions to compute rouge
        )

    def sim_scores(self, embeddings, trgt, mode='doc'):

        # use sim function depending on mode (util.cos_sim gives scores between each possible pair)
        sim_fn = F.cosine_similarity if mode=='doc' else util.cos_sim

        # compute similarity scores
        similarity = {
            ######## positive terms ######## 
            f'a, {trgt}': sim_fn(
                embeddings['a'],
                embeddings[trgt]
            ),
            f'b, {trgt}': sim_fn(
                embeddings['b'],
                embeddings[trgt]
            ),
            ################################ 
            ######## negative terms ######## 
            f'b, a-b': sim_fn(
                embeddings['b'],
                embeddings['a-b']
            ),
            f'a, b-a': sim_fn(
                embeddings['a'],
                embeddings['b-a']
            ),
            f'a-b, {trgt}': sim_fn(
                embeddings['a-b'],
                embeddings[trgt]
            ),
            f'b-a, {trgt}': sim_fn(
                embeddings['b-a'],
                embeddings[trgt]
            ),
            f'a-b, b-a': sim_fn(
                embeddings['a-b'],
                embeddings['b-a']
            ),
            ################################ 
        }
        
        return similarity


    def _ver1(
        self,
        batch: Dict,
        preds: torch.Tensor,
        action_space: int=2,
        k: int=3,
    ):

        dev = self.model.device

        ########################## unpack batch dictionary ########################## 

        intersection = batch['intersection']
        part_mask = batch['part_mask']
        seq_mask = batch['seq_mask']
        sent_mat = batch['sent_mat']

        ############################################################################# 
        
        if action_space == 2:
            # mask out irrelevant indices and convert to numpy
            bool_preds = preds.bool().cpu().numpy()
            bool_seq_mask = seq_mask.cpu().numpy()

            sent_mask = bool_preds * bool_seq_mask
            flipped_sent_mask = ~bool_preds * bool_seq_mask

        elif action_space == 4:
            sent_mask = self.build_mask(preds, seq_mask, k)
            breakpoint()

        # build set masks from predictions
        masks = {
            'intersect(a, b)': sent_mask,
            'union(a, b) - intersect(a, b)': flipped_sent_mask,
        }
        
        # build documents by indexing sent_mat using the masks
        docs = {key: [] for key in masks.keys()}
        make_doc = lambda doc, mask: ' '.join(doc[mask])
        for i, doc in enumerate(sent_mat):
            for key, mask in masks.items():
                docs[key].append(make_doc(doc, mask[i]))
        docs['true intersection'] = batch['intersection']

        # compute embeddings for each document set
        embeddings = {
            key: self.model.encode(
                doc_set,
                batch_size=self.batch_size,
                device=dev,
                convert_to_tensor=True,
            )
            for key, doc_set in docs.items()
        }

        # compute similarity scores
        scores = {
            'true intersection, intersect(a, b)': 
                F.cosine_similarity(
                    embeddings['true intersection'],
                    embeddings['intersect(a, b)']
                ),
            'true intersection, union(a, b) - intersect(a, b)': \
                F.cosine_similarity(
                    embeddings['true intersection'],
                    embeddings['union(a, b) - intersect(a, b)']
                ),
        }

        # compute reward
        G = (
            scores['true intersection, intersect(a, b)']
            - scores['true intersection, union(a, b) - intersect(a, b)']
        )

        # convert similarity keys for metric logging
        metrics = {
            f'reward/sim({key})': val.cpu().numpy() 
            for key, val in scores.items()
        }
        metrics['reward/sim(true intersection, union(a, b) - intersect(a, b))'] = \
            -metrics['reward/sim(true intersection, union(a, b) - intersect(a, b))']
        """
        metrics['reward/sim(true intersection, intersect(a, b))'] = \
            -metrics['reward/sim(true intersection, intersect(a, b))']
        """
        metrics['reward/aggregate'] = G.cpu().numpy()

        return (
            G.to(self.out_dev), # return reward for loss function
            metrics, # return sim scores for metric logging
            docs['intersect(a, b)'], # return document predictions to compute rouge
        )


