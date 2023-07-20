"""
this module is used to compute the SOFSAT reward for the AllSidesRanking environment
"""

# external imports
import torch
import numpy as np
import evaluate
from sentence_transformers import util
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score as ndcg_score_fn
from torch.nn import functional as F

# local imports
from mltoolkit.utils import (
    display
)
def ranking_reward(ep_info, encoder, K=3, dev='cpu'):

    # compute scores for each document
    #scores, top_doc = get_scores(ep_info, encoder, K, dev=dev)
    scores, top_doc = get_scores_rouge(ep_info, encoder, K, dev=dev)

    # compute correlation-based reward
    spearman_score =  spearmanr(
        scores,
        range(len(scores), 0, -1)
    )

    # compute ndcg score
    y_pred = np.zeros(len(scores))
    y_pred[np.argsort(scores)] = np.arange(len(scores))
    #y_pred = y_pred[::-1]

    y_true = np.arange(len(scores))[::-1]

    y_pred = y_pred[np.newaxis]
    y_true = y_true[np.newaxis]

    ndcg_score = ndcg_score_fn(y_true, y_pred, k=K)

    # sum spearman and ndcg scores
    reward = spearman_score.statistic + ndcg_score

    return (
        reward, 
        {
            'intersection': ep_info['intersection'],
            'top_doc': top_doc,
            'ndcg': ndcg_score,
            'spearman': spearman_score.statistic,
        }
    )

def get_scores(ep_info, encoder, K, dev='cpu'):

    ################ unpack vars ################ 

    a_emb = ep_info['a_emb']
    b_emb = ep_info['b_emb']

    len_a = ep_info['len_a']
    len_b = ep_info['len_b']
    L = len_a + len_b

    sentences = np.array(ep_info['sentences'])

    rankings = ep_info['rankings']
    if type(rankings) == list:
        rankings = np.array(rankings, dtype=int)
    else:
        rankings = rankings.astype(int)

    #############################################

    # filter out unsorted indices
    rankings = rankings[rankings != -1]

    # build a partition mask to map indices to their respective documents
    partition_mask = np.zeros(
        L,
        dtype=bool
    )
    partition_mask[len_a:] = True

    # build base document sets
    # a and b are the source documents
    # int_ab is the grouped set of documents based on the rankings (e.g. int_ab[0] = rankings[0:3])
    docs = {
        'a': ep_info['doc_a'],
        'b': ep_info['doc_b'],
        'int_ab': [
            ' '.join(sentences[rankings[i:i+K]]) 
            for i in range(len(rankings)-K+1)
        ],
    }

    N = len(docs['int_ab'])

    # make groups of K indices (corresponds with the docs variable)
    groups = np.array([
        rankings[i:i+K] 
        for i in range(len(rankings)-K+1)
    ])

    # create mask for the N document groups
    doc_mask = np.zeros((N, L), dtype=bool)
    X, Y = np.indices(doc_mask.shape)

    doc_mask[X[:, :K], groups] = True
    flipped_mask = ~doc_mask

    # add mask to dictionary and make the other masks
    masks = {
        'int_ab': doc_mask,
        'a-b': flipped_mask * (partition_mask[None] == False),
        'b-a': flipped_mask * (partition_mask[None] == True),
    }

    # build inferred sentence sets
    docs['a-b'] = [' '.join(sentences[mask]) for mask in masks['a-b']]
    docs['b-a'] = [' '.join(sentences[mask]) for mask in masks['b-a']]

    # compute embeddings for all sets
    embeddings = {
        'a': torch.tensor(
            ep_info['a_emb'],
            device=dev
        ),
        'b': torch.tensor(
            ep_info['b_emb'],
            device=dev
        ),
        'int_ab': encoder.encode(
            docs['int_ab'],
            convert_to_tensor=True,
            device=dev
        ),
        'a-b': encoder.encode(
            docs['a-b'],
            convert_to_tensor=True,
            device=dev
        ),
        'b-a': encoder.encode(
            docs['b-a'],
            convert_to_tensor=True,
            device=dev
        ),
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

    return scores.cpu().numpy(), docs['int_ab'][0]


def get_scores_rouge(ep_info, encoder, K, dev='cpu'):

    ################ unpack vars ################ 

    a_emb = ep_info['a_emb']
    b_emb = ep_info['b_emb']

    len_a = ep_info['len_a']
    len_b = ep_info['len_b']
    L = len_a + len_b

    sentences = np.array(ep_info['sentences'])

    rankings = ep_info['rankings']
    if type(rankings) == list:
        rankings = np.array(rankings, dtype=int)
    else:
        rankings = rankings.astype(int)

    #############################################

    # filter out unsorted indices
    rankings = rankings[rankings != -1]

    # build a partition mask to map indices to their respective documents
    partition_mask = np.zeros(
        L,
        dtype=bool
    )
    partition_mask[len_a:] = True

    # build base document sets
    # a and b are the source documents
    # int_ab is the grouped set of documents based on the rankings (e.g. int_ab[0] = rankings[0:3])
    docs = {
        'a': ep_info['doc_a'],
        'b': ep_info['doc_b'],
        'int_ab': sentences[rankings]
#        'int_ab': [
#            ' '.join(sentences[rankings[i:i+K]]) 
#            for i in range(len(rankings)-K+1)
#        ],
    }

    N = len(docs['int_ab'])


    # compute rouge
    rouge = evaluate.load('rouge')

    predictions = docs['int_ab']
    references = [ep_info['intersection']]*len(predictions)

    scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=False,
    )

    return np.array(scores['rouge1']), docs['int_ab'][0]

