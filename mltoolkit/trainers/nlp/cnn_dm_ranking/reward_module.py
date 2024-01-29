"""
this file contains code for the model used to compute rewards for the rl_extractive trainer
"""
# external imports
import evaluate
import re
import torch
import numpy as np
import pandas as pd
import itertools
from torch import nn
from torch.nn import functional as F
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import ndcg_score as ndcg_score_fn
from sklearn.metrics import dcg_score as dcg_score_fn
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


class RewardModule(nn.Module):
    
    def __init__(self, cfg):
        super(RewardModule, self).__init__()

        ######################## set variables from cfg #######################

        self.model = SentenceTransformer(
            cfg.model.get(
                'reward_model',
                'all-mpnet-base-v1'
            )
        )
        self.batch_size = cfg.model['reward_batch_size']
        self.out_dev = cfg.model['device']
        self.dev = cfg.model['reward_device']

        ######################################################################

    def forward(self, batch, rankings):

        #all_scores, docs = self.compute_scores(batch, rankings)
        all_scores, docs = self.compute_rouge(batch, rankings)

        ndcg_scores = []
        K=3
        num_buckets = 10
        samples_per_bucket = 3
        
        for scores in all_scores:

            buckets = self.make_buckets(
                scores,
                num_buckets=num_buckets,
                samples_per_bucket=2
            )

            # compute ndcg score
            y_true = np.zeros(len(scores))

            y_true[np.argsort(scores)] = buckets
            y_pred = buckets[::-1]

            y_pred = y_pred[np.newaxis]
            y_true = y_true[np.newaxis]

            ndcg_scores.append(ndcg_score_fn(y_true, y_pred))
            #ndcg_scores.append(ndcg_score_fn(y_true, y_pred, k=K))
            #ndcg_scores.append(dcg_score_fn(y_true, y_pred, k=K))
            #ndcg_scores.append(dcg_score_fn(y_true, y_pred))

        ndcg_scores = torch.tensor(ndcg_scores, device=self.out_dev)

        return (ndcg_scores, docs)

    def compute_rouge(self, batch, rankings):

        ########################## unpack batch dictionary ########################## 

        references = batch['intersection']
        partition_mask = batch['part_mask']
        seq_mask = batch['seq_mask'].cpu().numpy()
        sent_mat = batch['sent_mat']
        a_embs = batch['a_embs']
        b_embs = batch['b_embs']
        docs_a = batch['a']
        docs_b = batch['b']
        K = 3

        ############################################################################# 

        all_scores = []
        top_summaries = []

        # get indices for rearrangement
        X, _ = np.indices(rankings.shape)
        Y = np.fliplr(np.argsort(
            rankings.cpu().numpy()
        ))

        # rearrange sentences based on rankings
        ordered_sents = sent_mat[X, Y]
        ordered_masks = seq_mask[X, Y]

        all_scores = []
        top_summaries = []
        
        for reference, o_sents, o_mask, sents, mask, part_mask, doc_a, doc_b, a_emb, b_emb, Xi, Yi  in zip(
            references,
            ordered_sents,
            ordered_masks,
            sent_mat,
            seq_mask,
            partition_mask,
            docs_a,
            docs_b,
            a_embs,
            b_embs,
            X,
            Y
        ):
           
            o_sents = o_sents[o_mask]
            Yi = Yi[o_mask]

            #preds = [' '.join(o_sents[i:i+K]) for i in range(len(o_sents)-K+1)]
            #preds = [' '.join(o_sents[i:i+K]) for i in range(0, len(o_sents), 3)]
            preds = [sent for sent in o_sents]
            refs = [reference]*len(preds)

            rouge_scorer = evaluate.load('rouge')

            scores = rouge_scorer.compute(
                predictions=preds,
                references=refs,
                use_aggregator=False,
            )

            scores = scores['rouge1']

            all_scores.append(np.array(scores))
            top_summaries.append(preds[0])

        return all_scores, top_summaries



    def compute_scores(self, batch, rankings):

        ########################## unpack batch dictionary ########################## 

        intersection = batch['intersection']
        partition_mask = batch['part_mask']
        seq_mask = batch['seq_mask'].cpu().numpy()
        sent_mat = batch['sent_mat']
        a_embs = batch['a_embs']
        b_embs = batch['b_embs']
        docs_a = batch['a']
        docs_b = batch['b']
        K = 3

        ############################################################################# 

        all_scores = []
        top_summaries = []

        # get indices for rearrangement
        X, _ = np.indices(rankings.shape)
        Y = np.fliplr(np.argsort(
            rankings.cpu().numpy()
        ))

        # rearrange sentences based on rankings
        ordered_sents = sent_mat[X, Y]
        ordered_masks = seq_mask[X, Y]
        
        for o_sents, o_mask, sents, mask, part_mask, doc_a, doc_b, a_emb, b_emb, Xi, Yi  in zip(
            ordered_sents,
            ordered_masks,
            sent_mat,
            seq_mask,
            partition_mask,
            docs_a,
            docs_b,
            a_embs,
            b_embs,
            X,
            Y
        ):
            o_sents = o_sents[o_mask]
            Yi = Yi[o_mask]

            docs = {
                'a': doc_a,
                'b': doc_b,
                'int_ab': [
                    ' '.join(o_sents[i:i+K]) 
                    for i in range(len(o_sents)-K+1)
                ],
            }

            L = len(sents)
            N = len(docs['int_ab'])

            # create mask for the N document groups
            group_mask = np.zeros((N, L), dtype=bool)

            groups_x = np.arange(N)[None].repeat(K, axis=0).T
            groups_y = sliding_window_view(Yi, K)
            
            group_mask[groups_x, groups_y] = True
            flipped_mask = ~group_mask

            # add mask to dictionary and make the other masks
            set_masks = {
                #'int_ab': doc_mask,
                'a-b': flipped_mask * (part_mask == 1),
                'b-a': flipped_mask * (part_mask == 2),
            }

            # build inferred sentence sets
            docs['a-b'] = [' '.join(sents[set_mask]) for set_mask in set_masks['a-b']]
            docs['b-a'] = [' '.join(sents[set_mask]) for set_mask in set_masks['b-a']]

            # get encodings
            grouped_embeddings = self.model.encode(
                docs['int_ab'] + docs['a-b'] + docs['b-a'],
                convert_to_tensor=True,
                batch_size=self.batch_size,
                device=self.dev
            )

            _, E = grouped_embeddings.shape
            grouped_embeddings = grouped_embeddings.reshape((N, -1, E))
            grouped_embeddings = grouped_embeddings.transpose(0, 1)

            embeddings = {
                'a': a_emb,
                'b': b_emb,
                'int_ab': grouped_embeddings[0],
                'a-b': grouped_embeddings[1],
                'b-a': grouped_embeddings[2],
            }

            # get similarity scores for all sets
            sim_scores = {
                ######## positive terms ######## 
                'a, int_ab': F.cosine_similarity(
                    embeddings['a'],
                    embeddings['int_ab']
                ),
                'b, int_ab': F.cosine_similarity(
                    embeddings['b'],
                    embeddings['int_ab']
                ),
                ################################ 
                ######## negative terms ######## 
                'b, a-b': F.cosine_similarity(
                    embeddings['b'],
                    embeddings['a-b']
                ),
                'a, b-a': F.cosine_similarity(
                    embeddings['a'],
                    embeddings['b-a']
                ),
                'a-b, int_ab': F.cosine_similarity(
                    embeddings['a-b'],
                    embeddings['int_ab']
                ),
                'b-a, int_ab': F.cosine_similarity(
                    embeddings['b-a'],
                    embeddings['int_ab']
                ),
                'a-b, b-a': F.cosine_similarity(
                    embeddings['a-b'],
                    embeddings['b-a']
                ),
                ################################ 
            }

            # compute aggregate scores
            scores = (
                sim_scores['a, int_ab']
                + sim_scores['b, int_ab']
                - sim_scores['a, b-a']
                - sim_scores['b, a-b']
                - sim_scores['a-b, int_ab']
                - sim_scores['b-a, int_ab']
                - sim_scores['a-b, b-a']
            )

            all_scores.append(scores.cpu().numpy())
            top_summaries.append(docs['int_ab'][0])

        return all_scores, top_summaries

    def make_buckets(self, scores, num_buckets=10, samples_per_bucket=2):

        tail_buckets = np.repeat(np.arange(num_buckets), samples_per_bucket)

        if len(tail_buckets) > len(scores):
            return tail_buckets[-len(scores):]

        head_buckets = np.zeros(len(scores) - len(tail_buckets))

        return np.concatenate((
            head_buckets,
            tail_buckets
        ))
     
