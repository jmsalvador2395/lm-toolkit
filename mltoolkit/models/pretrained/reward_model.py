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
        if self.procedure == 'rl_extractive':
            return self.rl_extractive(batch, preds)
        elif self.procedure == 'sofsat_extractive_sum':
            return self.sofsat_extractive_sum(batch, preds)

    def evaluate_episodes(self, batch, preds):
        return self.sofsat_extractive_sum(batch, preds)

    def sofsat_extractive_sum(self, batch: Dict, preds: torch.Tensor):

        dev = self.model.device

        ########################## unpack batch dictionary ########################## 

        intersection = batch['intersection']
        part_mask = batch['part_mask']
        seq_mask = batch['seq_mask'].cpu().numpy()
        sent_mat = batch['sent_mat']

        ############################################################################# 
        
        # mask out irrelevant indices and convert to numpy
        preds = seq_mask * preds.bool().cpu().numpy()
        flipped_preds = ~preds

        # build set masks from predictions
        masks = {
            'intersect(a, b)': preds,
            'a-b': flipped_preds * (batch['part_mask'] == 1),
            'b-a': flipped_preds * (batch['part_mask'] == 2),
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

        # compute similarity scores
        similarity = {
            ######## positive terms ######## 
            'a, intersect(a, b)': F.cosine_similarity(
                embeddings['a'],
                embeddings['intersect(a, b)']
            ),
            'b, intersect(a, b)': F.cosine_similarity(
                embeddings['b'],
                embeddings['intersect(a, b)']
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
            'a-b, intersect(a, b)': F.cosine_similarity(
                embeddings['a-b'],
                embeddings['intersect(a, b)']
            ),
            'b-a, intersect(a, b)': F.cosine_similarity(
                embeddings['b-a'],
                embeddings['intersect(a, b)']
            ),
            'a-b, b-a': F.cosine_similarity(
                embeddings['a-b'],
                embeddings['b-a']
            ),
            ################################ 
        }
        
        # compute reward
        G = (
            similarity['a, intersect(a, b)']
            + similarity['b, intersect(a, b)']
            - similarity['a, b-a']
            - similarity['b, a-b']
            - similarity['a-b, intersect(a, b)']
            - similarity['b-a, intersect(a, b)']
            - similarity['a-b, b-a']
        )

        # convert similarity keys for metric logging
        metrics= {
            f'reward/sim({key})': val.cpu().numpy() 
            for key, val in similarity.items()
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


    def rl_extractive(self, batch, preds):
        sent_mat, a, b, len_df, a_minus_b, a_int_b, b_minus_a = \
            self._get_three_partitions(batch, preds.detach().cpu().numpy())
        return (
            sent_mat,
            self._compute_reward(
                sent_mat,
                a,
                b,
                a_minus_b,
                a_int_b,
                b_minus_a
            )
        )


    def _get_three_partitions(self, batch, pred_y: np.ndarray) -> Tuple:
        """
        Given the model prediction (1/0 for each sentence), divide into three sets. A-B, A \int B, B-A
        :param batch: batch of data
        :param pred_y: here pred_y acts as a selection map for sentences in intersection set
        :return:
        """
        batch_sz, max_sents = pred_y.shape

        # combined sent mat
        doc_names = list(batch.keys())  # {s1, s2} in any order
        comb_src = {doc: data["src_sents"] for doc, data in batch.items()}
        # aa = [_add_full_stop_at_end(doc_sents) for doc_sents in comb_src[doc_names[0]]]
        # bb = [ii[0] for ii in aa]
        # cc = [ii[0] for ii in comb_src[doc_names[0]]]
        # if not all([x == y for x, y in zip(bb, cc)]):
        #     zz = 1
        # zz = 1
        for key in comb_src.keys():
            comb_src[key] = [
                _add_full_stop_at_end(doc_sents) for doc_sents in comb_src[key]
            ]
        sent_mat, len_df = _src_sent_to_sent_mat(comb_src)

        # get set A
        d_name = doc_names[0]
        a_sent_mat, a_len = _src_sent_to_sent_mat({d_name: batch[d_name]["src_sents"]})
        assert (
            a_len.columns[0] == len_df.columns[0]
        ), "Make sure there is no issue of ordering when combining docs"

        # get set B
        d_name = doc_names[1]
        b_sent_mat, b_len = _src_sent_to_sent_mat({d_name: batch[d_name]["src_sents"]})
        assert (
            b_len.columns[0] == len_df.columns[1]
        ), "Make sure there is no issue of ordering when combining docs"

        # sentences in intersection set
        a_int_b = _select_sentences(sent_mat, pred_y)

        invert_pred_y = (pred_y < 0.5).astype(
            int
        )  # we want the sentences not in intersection
        # sentences in A - B set
        ab_map = (
            np.array(
                [
                    [1] * row[0] + [0] * row[1] + [0] * (max_sents - row[2])
                    for idx, row in len_df.iterrows()
                ]
            )
            * invert_pred_y
        )
        a_minus_b = _select_sentences(sent_mat, ab_map)

        # sentences in B - A set
        ba_map = (
            np.array(
                [
                    [0] * row[0] + [1] * row[1] + [0] * (max_sents - row[2])
                    for idx, row in len_df.iterrows()
                ]
            )
            * invert_pred_y
        )
        b_minus_a = _select_sentences(sent_mat, ba_map)

        return sent_mat, a_sent_mat, b_sent_mat, len_df, a_minus_b, a_int_b, b_minus_a



    def _compute_reward(
        self,
        sent_mat,
        a_sent_mat,
        b_sent_mat,
        a_minus_b,
        a_int_b,
        b_minus_a
    ) -> torch.Tensor:
        """Computes the reward scores"""

        device = self.model.device

        batch_sz = sent_mat.shape[0]
        doc_rewards = {}
        sent_rewards = {}

        # Reward equation I (Negative reward Terms)
        partition_dict = {
            "a_minus_b": a_minus_b,
            "a_int_b": a_int_b,
            "b_minus_a": b_minus_a,
        }

        for comb in itertools.combinations(partition_dict.keys(), 2):
            new_key = "__".join(comb) + "__neg"
            doc_rewards[f"doc__{new_key}"] = self._doc_level_reward(
                *tuple(map(lambda key: partition_dict[key], comb)),
                batch_sz,
                device
            )

            sent_rewards[f"sent__{new_key}"] = self._sent_level_reward(
                *tuple(map(lambda key: partition_dict[key], comb)),
                batch_sz,
                device
            )

        # Reward equation II (Positive reward Terms)
        doc_rewards["doc__a__a_int_b__pos"] = \
            self._doc_level_reward(
                a_sent_mat,
                a_int_b,
                batch_sz,
                device=device
            )
        doc_rewards["doc__b__a_int_b__pos"] = \
            self._doc_level_reward(
                b_sent_mat,
                a_int_b,
                batch_sz,
                device=device
            )

        sent_rewards["sent__a__a_int_b__pos"] = \
            self._sent_level_reward(
                a_sent_mat,
                a_int_b,
                batch_sz,
                device=device
            )
        sent_rewards["sent__b__a_int_b__pos"] = \
            self._sent_level_reward(
                b_sent_mat,
                a_int_b,
                batch_sz,
                device=device
            )

        # Reward equation III (Negative reward Terms)
        doc_rewards["doc__a__b_minus_a__neg"] = \
            self._doc_level_reward(
                a_sent_mat,
                b_minus_a,
                batch_sz,
                device=device
            )
        doc_rewards["doc__b__a_minus_b__neg"] = \
            self._doc_level_reward(
                b_sent_mat,
                a_minus_b,
                batch_sz,
                device=device
            )

        sent_rewards["sent__a__b_minus_a__neg"] = \
            self._sent_level_reward(
                a_sent_mat,
                b_minus_a,
                batch_sz,
                device=device
            )
        sent_rewards["sent__b__a_minus_b__neg"] = \
            self._sent_level_reward(
                b_sent_mat,
                a_minus_b,
                batch_sz,
                device=device
            )

        # Best Weight Combination based on Pearson Correlation Results
        # Custom weights
        # Note: I have added a big constant to make everything positive. Range is [2, -2]
        return 0.25 * (  # doc
            0.75  # doc_pos
            * (
                0.25 * doc_rewards["doc__a__a_int_b__pos"]
                + 0.75 * doc_rewards["doc__b__a_int_b__pos"]
            )
            + 0.25  # doc_neg
            * (-1)
            * (
                0.1 * doc_rewards["doc__a_minus_b__a_int_b__neg"]
                + 0.0 * doc_rewards["doc__a_minus_b__b_minus_a__neg"]
                + 0.9 * doc_rewards["doc__a_int_b__b_minus_a__neg"]
                + 0.0 * doc_rewards["doc__a__b_minus_a__neg"]
                + 0.0 * doc_rewards["doc__b__a_minus_b__neg"]
            )
        ) + 0.75 * (  # sent
            0.7  # sent_pos  # TODO: try high value of this
            * (
                0.5 * sent_rewards["sent__a__a_int_b__pos"]
                + 0.5 * sent_rewards["sent__b__a_int_b__pos"]
            )
            + 0.3  # sent_neg  # 0.5
            * (-1)
            * (
                0.0 * sent_rewards["sent__a_minus_b__a_int_b__neg"]
                + 0.1 * sent_rewards["sent__a_minus_b__b_minus_a__neg"]
                + 0.0 * sent_rewards["sent__a_int_b__b_minus_a__neg"]
                + 0.8 * sent_rewards["sent__a__b_minus_a__neg"]
                + 0.1 * sent_rewards["sent__b__a_minus_b__neg"]
            )
        )

    # #####################################################
    # Document level reward
    # #####################################################
    def _doc_level_reward(
        self,
        partition1: np.ndarray,
        partition2: np.ndarray,
        batch_sz: int,
        device: str='cpu',
    ) -> torch.Tensor:
        def convert_to_document(sel_sent_mat: np.ndarray) -> list:
            return [" ".join((" ".join(ii)).strip().split()) for ii in sel_sent_mat]

        # Convert sentence matrix to list
        part1_docs, part2_docs = convert_to_document(
            partition1
        ), convert_to_document(partition2)
        assert len(part1_docs) == len(part2_docs), "Batch length mismatch. Check"

        # encode sentence matrix
        flatten_docs = part1_docs + part2_docs
        flatten_embeds = self.model.encode(
            flatten_docs,
            batch_size=self.batch_size,
            device=device,
            convert_to_tensor=True,
        )
        part1_embeds, part2_embeds = torch.split(flatten_embeds, batch_sz, dim=0)
        assert (
            part1_embeds.shape == part2_embeds.shape
        ), "Size mismatch in doc reward computation"

        # create mask to handle empty docs
        mask = torch.tensor(
            list(map(lambda x: 1 if x else 0, flatten_docs)), device=device
        )
        mask1, mask2 = torch.split(mask, batch_sz, dim=0)
        mask = (
            mask1 * mask2
        )  # if any doc is empty string it would make the similarity 0

        # cosine similarity
        return (
            nn.functional.cosine_similarity(part1_embeds, part2_embeds, dim=1)
            * mask
        ).cpu()

    # #####################################################
    # Sent level reward
    def _sent_level_reward(
        self,
        partition1: np.ndarray,
        partition2: np.ndarray,
        batch_sz,
        device: str='cpu'
    ):

        part1_count, part2_count = _get_sent_count_in_each_partition(
            (partition1, partition2)
        )
        # flatten partitions (raterization, first across cols (for a sample), then across rows (batch))
        flatten_sents = (
            partition1[partition1.nonzero()].tolist()
            + partition2[partition2.nonzero()].tolist()
        )
        # get embeddings
        if len(flatten_sents) > 0:  # atleast one sentence in two partitions
            flatten_embeds = self.model.encode(
                flatten_sents,
                batch_size=self.batch_size,
                device=device,
                convert_to_tensor=True,
            )
            # split returns tuple of tensors
            part1_embeds, part2_embeds = torch.split(
                flatten_embeds, [sum(part1_count), sum(part2_count)], dim=0
            )

            # split for sentence in each sample doc in batch (derastrize in a way)
            part1_sent_embeds: tuple = torch.split(
                part1_embeds, part1_count.tolist(), dim=0
            )
            part2_sent_embeds: tuple = torch.split(
                part2_embeds, part2_count.tolist(), dim=0
            )

            # Note: no need to create mask here since that is taken care by the split function itself.
            #       For empty doc, it returns tensor fo shape [0, 768]

            # cosine similarity  (indexing over batch dim here)
            # by default torch tensor is created on the cpu
            return torch.tensor(
                [
                    _compute_mean(util.cos_sim(ii, jj))
                    for ii, jj in zip(part1_sent_embeds, part2_sent_embeds)
                ]
            )
        else:  # literally no sentence in both the partitions
            return torch.zeros(batch_sz)  # return 0 similarity vec of length batch


def _compute_mean(sim_tensor: torch.Tensor) -> float:
    # mean for empty tensor returns nan. Set it to 0
    return (
        0.0 if sim_tensor.nelement() == 0 else torch.mean(sim_tensor).item()
    )

# Returns non-empty sentences count across batch
def _get_sent_count_in_each_partition(partitions: tuple) -> list:
    out = []
    for part in partitions:
        non_zeros = part.nonzero()[
            0
        ]  # since we only want to count non-empty str across cols
        # in case some rows have all zeros, that won't show up. So explicitly start from 0
        tmp = np.zeros(len(part), dtype=int)
        eles, cts = np.unique(non_zeros, return_counts=True)
        tmp[eles] = cts
        out.append(tmp)
    return out

def _select_sentences(sentence_matrix, bool_map: np.ndarray) -> np.ndarray:
    assert (
        sentence_matrix.shape == bool_map.shape
    ), "Shape mismatch while selecting sentences"
    return np.char.multiply(sentence_matrix, bool_map)
    # return [' '.join((" ".join(ii)).strip().split()) for ii in sel_sent_mat]

# Coovert src sentences to sentence matrix
def _src_sent_to_sent_mat(
    src_dict: Dict[str, list]
) -> Tuple[np.ndarray, pd.DataFrame]:
    df = pd.DataFrame.from_dict(src_dict)
    len_df = df.applymap(len)  # also a data frame
    # len_df.columns = [f'{ii}_len' for ii in len_df.columns]
    len_df["total_len"] = len_df.sum(axis=1)
    max_tot_sents = len_df["total_len"].max()
    df["empty_lists"] = len_df.apply(
        lambda row: (max_tot_sents - row.total_len) * [""], axis=1
    )
    return (
        np.array(
            df.apply(
                lambda row: sum(
                    row.to_list(), []
                ),  # lambda func flattens list of lists here
                axis=1,
            ).to_list()
        ),
        len_df,
    )

def _add_full_stop_at_end(doc_sents: List[str]):
    # TODO: not correct. Pick the last non empty sentence
    regex = re.compile("[.,@_!#$%^&*()<>?/\|}{~:]")
    last_sent = doc_sents[-1].strip()
    if doc_sents[-1] and regex.search(last_sent[-1]) == None:
        last_sent += "."
    doc_sents[-1] = last_sent
    return doc_sents


