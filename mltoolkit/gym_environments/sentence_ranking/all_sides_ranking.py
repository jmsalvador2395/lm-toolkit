"""
this file contains the gym environment for extractive summarization with ranking
"""
# external imports
import numpy as np
import torch
import gymnasium as gym
from datasets import Dataset
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from itertools import product, zip_longest
from argparse import Namespace

# local imports
from .reward_module import ranking_reward
from mltoolkit.utils import (
    display
)

class AllSidesRankingEnv(gym.Env):

    def __init__(self, cfg: Dict):

        obs_types = {
            'torch': torch.float32,
            'numpy': np.float32,
        }

        ################ Check Config Values ################
        
        if cfg.data['obs_type'] not in ['torch', 'numpy']:
            display.error(
                'invalid observation type. '
                + 'valid types: [\'torch\', \'numpy\']'
            )
            raise ValueError()

        if cfg.data['render_mode'] is not None:
            display.error(
                'invalid render mode. supported: None'
            )
            raise ValueError()

        #####################################################

        ############### set initial parameters #############

        self.mode = 'train'
        self.epoch = 0
        self.obs_type = cfg.data['obs_type']
        self.dtype = obs_types[self.obs_type]
        self.render_mode = None
        self.epoch=0
        self.epoch_reset = False
        self.ordering_method = cfg.data['ordering_method']

        ####################################################

        ################ Set Fixed Values ################ 

        self.obs_shape = (768*4,)
        self.n = 2
        self.action_space = gym.spaces.Discrete(
            self.n,
            seed=cfg.general['seed'],
        )
        self.observation_space = gym.spaces.Box(
            low=np.full(self.obs_shape, float('-inf')),
            high=np.full(self.obs_shape, float('inf')),
            shape=self.obs_shape,
            dtype=np.float32,
            seed=cfg.general['seed'],
        )
        self.reward_range = (-1, 1)

        ##################################################

        ############### set cfg-based vars ############### 

        self.ds = self._setup_data(cfg)
        self.emb_dev = cfg.data['embedding_device']
        emb_model_name = cfg.data['embedding_model']
        self.emb_model = SentenceTransformer(
            emb_model_name,
            device=self.emb_dev
        )
        self.emb_batch_size = cfg.data['embedder_batch_size']
        self.K = cfg.data['doc_group_size']
        self.method = cfg.data['method']

        ##################################################

    def render(self):
        pass

    def _setup_data(self, cfg):
        """
        sets up the dataset that will be used to build the environment episodes

        Input
            cfg[Dict]: the config dictionary

        Return
            ds[datasets.DatasetDict]: the source dataset
        """

        # initialize dataset
        ds = Dataset.from_csv(cfg.data['loc'])

        # TODO delete this (testing on subset of the dataset)
        ds = ds.select(range(3))

        # truncate dataset columns
        trgt_columns = {
            "left-context": "doc_a",
            "right-context": "doc_b",
            "theme-description": "intersection",
        }
        ds = ds.remove_columns(
            set(ds.features.keys())
            - set(trgt_columns.keys())
        )
        ds = ds.rename_columns(trgt_columns)

        # split into train and test sets
        ds = ds.train_test_split(
            test_size=1-cfg.data['train_test_split'],
            train_size=cfg.data['train_test_split'],
        )

        # shuffle the data indices
        self.train_order = np.arange(
            len(ds['train'])
        )
        self.eval_order = np.arange(
            len(ds['test'])
        )
        np.random.shuffle(self.train_order)
        np.random.shuffle(self.eval_order)

        self.obs_type = cfg.data['obs_type']

        return ds

    def reset(
        self,
        seed: int | None=None,
        options: Dict[str, Any]={},
    ) -> Tuple:

        full_reset = options.get('full_reset', False)
        if full_reset or self.epoch_reset:

            # reset the epoch flag
            self.epoch_reset = False

            # set counters
            self.order_idx = 0
            self.epoch = 0 if full_reset else self.epoch+1

        # go to next sample if resets aren't called
        else:
            self.order_idx += 1

        # build vectors for current episode
        s0, self.ep_info = self._setup_episode(
            self.train_order[self.order_idx],
        )

        return (s0, self.ep_info)

    def _setup_episode(self, idx, mode='train'):
        if self.method == 'naive':
            return self.naive_episode_setup(idx, mode)
        elif self.method == 'insertion_sort':
            return self.isort_episode_setup(idx, mode)
        else:
            raise ValueError()
            
    def isort_episode_setup(self, idx, mode='train'):
        """
        used in reset to prepare the episode vectors
        """

        ########### unpack sample ########### 

        sample = self.ds['train'][int(idx)]

        intersection = sample['intersection']
        doc_a = sample['doc_a']
        doc_b = sample['doc_b']

        #####################################

        # initialize dictionary of episode information
        ep_info = sample

        """
        # randomly swap documents 
        swap = np.random.randint(2)
        if swap:
            doc_a, doc_b = doc_b, doc_a
        """

        sentences_a = sent_tokenize(doc_a)
        sentences_b = sent_tokenize(doc_b)
        len_a = len(sentences_a)
        len_b = len(sentences_b)
        sentences = sentences_a + sentences_b

        if len_a + len_b <= 1:
            display.error('Not enough sentences for ranking')
            raise ValueError()

        # aggregate all sentences and documents before encoding
        all_text = [intersection, doc_a, doc_b] + sentences

        # encode all relevant text
        embs = self.emb_model.encode(
            all_text,
            batch_size=self.emb_batch_size,
            device=self.emb_dev,
        )

        # unpack embeddings into dictionary
        ep_info.update({
            'doc_a': doc_a,
            'doc_b': doc_b,
            'len_a': len_a,
            'len_b': len_b,
            'sentences': sentences,
            'intersection_emb': embs[0],
            'a_emb': embs[1],
            'b_emb': embs[2],
            'sentence_embs': embs[3:],
        })

        # create interleave indices for ranking order (ranking episode alternates between doc a and doc b.
        #indices = list(range(len_a + len_b))
        indices = self.get_ordering(len_a, len_b)

        # initialize variables for ranking (add first element to rankings)
        ep_info['rankings'] = indices[:1]
        ep_info['to_rank'] = indices[1:]
        ep_info['target'] = ep_info['to_rank'].pop(0)
        ep_info['target_rank'] = 0
        ep_info['t'] = 0

        pair0 = (
            ep_info['target'],
            ep_info['rankings'][ep_info['target_rank']]
        )

        s0 = np.hstack((
            ep_info['a_emb'],
            ep_info['b_emb'],
            ep_info['sentence_embs'][pair0[0]],
            ep_info['sentence_embs'][pair0[1]],
        ))
        
        return s0, ep_info

    def get_ordering(self, len_a, len_b):

        method = self.ordering_method
        methods = [
            'basic',
            'alternating',
            'alternating+',
        ]
        max_len = max(len_a, len_b)

        if method == 'basic':
            return list(range(len_a + len_b))

        elif method == 'random':
            indices = np.arange(len_a + len_b)
            np.random.shuffle(indices)
            return list(indices)

        elif method == 'alternating':

            indices = -np.ones((max_len, 2))

            a_indices = np.arange(len_a)
            b_indices = np.arange(len_a, len_a+len_b)

            indices[:len_a, 0] = a_indices
            indices[:len_b, 1] = b_indices

            #np.random.shuffle(a_indices)
            #np.random.shuffle(b_indices)

            indices = indices.reshape((-1,))
            indices = indices[indices != -1]
            indices = list(indices.astype(int))

            return indices
        elif method == 'alternating+':

            a_indices = np.arange(len_a)
            a_front = a_indices[:len_a//2]
            a_back = a_indices[len_a//2:]

            b_indices = np.arange(len_a, len_a+len_b)
            b_front = b_indices[:len_b//2]
            b_back = b_indices[len_b//2:]

            indices = np.hstack(
                list(zip_longest(
                    a_front,
                    a_back,
                    b_front,
                    b_back,
                    fillvalue=-1
                ))
            )

            indices = list(indices[indices != -1])
            return indices

        else:
            display.error('ordering method is invalid.')
            raise ValueError()

    def naive_episode_setup(self, idx, mode='train'):
        """
        used in reset to prepare the episode vectors
        """

        ########### unpack sample ########### 

        sample = self.ds['train'][int(idx)]

        intersection = sample['intersection']
        doc_a = sample['doc_a']
        doc_b = sample['doc_b']

        #####################################

        # initialize dictionary of episode information
        ep_info = sample

        # randomly swap documents 
        swap = np.random.randint(2)
        if swap:
            doc_a, doc_b = doc_b, doc_a

        sentences_a = sent_tokenize(doc_a)
        sentences_b = sent_tokenize(doc_b)
        len_a = len(sentences_a)
        len_b = len(sentences_b)
        sentences = sentences_a + sentences_b

        # aggregate all sentences and documents before encoding
        all_text = [intersection, doc_a, doc_b] + sentences

        # encode all relevant text
        embs = self.emb_model.encode(
            all_text,
            batch_size=self.emb_batch_size,
            device=self.emb_dev,
        )

        # unpack embeddings into dictionary
        ep_info.update({
            'doc_a': doc_a,
            'doc_b': doc_b,
            'len_a': len_a,
            'len_b': len_b,
            'sentences': sentences,
            'intersection_emb': embs[0],
            'a_emb': embs[1],
            'b_emb': embs[2],
            'sentence_embs': embs[3:],
        })

        # initialize variables for ranking
        union_len = len(ep_info['sentence_embs'])
        ep_info['ranking_seq'] = list(product([0], range(1, union_len)))
        ep_info['t'] = 0
        ep_info['T'] = len(ep_info['ranking_seq'])
        ep_info['rankings'] = -np.ones(len(ep_info['sentence_embs']))

        # build s0
        pair0 = ep_info['ranking_seq'][0]
        s0 = np.hstack((
            ep_info['a_emb'],
            ep_info['b_emb'],
            ep_info['sentence_embs'][pair0[0]],
            ep_info['sentence_embs'][pair0[1]],
        ))

        # these variables are used for keeping track of the ranking procedure
        ep_info['target'] = pair0[0]
        ep_info['less_than_target'] = []
        ep_info['greater_than_target'] = []
        ep_info['ranking_groups'] = []
        ep_info['rank_offset'] = 0
        ep_info['epoch'] = self.epoch

        return s0, ep_info

    def step(self, action: int):
        if self.method == 'naive':
            return self.naive_step(action)
        elif self.method == 'insertion_sort':
            return self.isort_step(action)
        else:
            raise ValueError()

    def isort_step(self, action: int):

        ########### initialize return values ########### 

        state = np.zeros(768*4)
        reward = 0
        terminated = False
        trunc = False
        info = {
            'end_of_epoch': False,
        }

        ################################################

        ########### unpack episode vars ########### 

        ep_info = self.ep_info

        target = ep_info['target']
        target_rank = ep_info['target_rank']
        to_rank = ep_info['to_rank']
        rankings = ep_info['rankings']

        ############################################

        # set flag to prepare next target
        prepare_next_target = False
        reward_flag = False

        # action for saying target is greater than sentence at target_rank
        if action == 0:
            
            # insert target into ranked position and set flag to assign next target
            self.ep_info['rankings'].insert(target_rank, target)
            prepare_next_target = True

            #reward_flag = True if len(self.ep_info['rankings']) > self.K else False
               
        elif action == 1:
            
            # increment ranking
            target_rank += 1
            ep_info['target_rank'] = target_rank

            # check if target_rank is at the end of the rankings list
            if target_rank == len(ep_info['rankings']):
                ep_info['rankings'].insert(target_rank, target)
                prepare_next_target = True

                #reward_flag = True if len(self.ep_info['rankings']) > self.K else False

        if prepare_next_target:

            # set terminated variable
            terminated = (len(to_rank) == 0)

            if terminated:
                # check for end of epoch
                if self.order_idx == len(self.ds[self.mode])-1:
                    info['end_of_epoch'] = True
                    self.epoch_reset = True

            else:
                # update ep_info
                ep_info['target_rank'] = 0

                target = ep_info['to_rank'].pop(0)
                ep_info['target'] = target

        if not terminated:

            # get sentence embeddings
            s1 = ep_info['sentence_embs'][target]
            s2 = ep_info['sentence_embs'][rankings[target_rank]]

            # get next state
            state = np.concatenate((
                ep_info['a_emb'],
                ep_info['b_emb'],
                s1,
                s2,
            ))

        else:
            reward_flag = True

        if reward_flag:
            reward, result = self._compute_reward(self.ep_info)
            info.update(result)

        return (
            state,
            reward,
            terminated,
            trunc,
            info,
        )

    def naive_step(self, action: int):

        ########### initialize return values ########### 

        state = None
        reward = 0
        terminated = False
        trunc = False
        info = {
            'end_of_epoch': False,
        }

        ################################################

        # set alias for class vars
        ep_info = self.ep_info

        # unpack vars
        t = ep_info['t']
        T = ep_info['T']
        candidate = ep_info['ranking_seq'][t][1]
        target = ep_info['target']

        if action == 0:
            ep_info['less_than_target'].append(candidate)
        elif action == 1:
            ep_info['greater_than_target'].append(candidate)
        else:
            display.error(f'action \"{action}\" is not an element of the action space')
            raise ValueError()

        # update timestep counter
        t += 1
        ep_info['t'] = t

        # prepare next round of ranking or end episode if time step 't' has reached time horizon T
        if t == T:

            offset = ep_info['rank_offset']
            
            # determine global rank of target
            global_rank = len(ep_info['greater_than_target']) + offset
            ep_info['rankings'][global_rank] = target

            # append next set of ranking groups
            left_group = {
                'rank_offset': offset,
                'ids': ep_info['greater_than_target']
            }
            right_group = {
                'rank_offset': global_rank + 1,
                'ids': ep_info['less_than_target']
            }

            # handle special case len(group) == 1
            if len(left_group['ids']) == 1:
                l_offset, l_ids = left_group.values()
                ep_info['rankings'][l_offset] = l_ids[0]
                left_group = {}

            if len(right_group['ids']) == 1: 
                r_offset, r_ids = right_group.values()
                ep_info['rankings'][r_offset] = r_ids[0]
                right_group = {}

            ep_info['ranking_groups'] += [
                left_group,
                right_group
            ]

            # filter out empty ranking groups
            ep_info['ranking_groups'] = list(filter(
                lambda grp: len(grp.get('ids', [])) != 0,
                ep_info['ranking_groups']
            ))

            """
            # compute reward if there are enough samples to compute rankings
            if np.sum(ep_info['rankings'] != -1) > self.K:
                reward = self._compute_reward(self.ep_info)
            """
            
            # update ep_info and prepares next ranking batch
            if len(ep_info['ranking_groups']) != 0:
               
                # unpack next first group in ranking_groups
                next_group = ep_info['ranking_groups'].pop(0)
                next_offset = next_group['rank_offset']
                next_ids = next_group['ids']

                # compute next round of pairings
                next_pairs = list(
                    product(next_ids[:1], next_ids[1:])
                )

                # update ep_info dictionary
                ep_info['ranking_seq'] += next_pairs
                ep_info['target'] = ep_info['ranking_seq'][t][0]
                ep_info['rank_offset'] = next_offset
                ep_info['T'] = len(ep_info['ranking_seq'])

                ep_info['less_than_target'] = []
                ep_info['greater_than_target'] = []

                # get sentence embeddings
                s1_idx, s2_idx = ep_info['ranking_seq'][t]
                s1 = ep_info['sentence_embs'][s1_idx]
                s2 = ep_info['sentence_embs'][s2_idx]

                # get next state
                state = np.concatenate((
                    ep_info['a_emb'],
                    ep_info['b_emb'],
                    s1,
                    s2,
                ))

            # goes here if episode is done
            else:

                # check if there are unfilled ranks
                rankings = ep_info['rankings']
                if -1 in rankings:
                    display.error('ranking procedure failed')
                    raise ValueError()

                # check if there are duplicate rankings
                if len(rankings) != len(set(rankings)):
                    display.error('duplicates exist in rankings array')
                    raise ValueError()

                # set termination flag
                state = np.zeros(768*4)
                terminated = True
                reward = self._compute_reward(self.ep_info)

                # check for end of epoch
                if self.order_idx == len(self.ds[self.mode])-1:
                    info['end_of_epoch'] = True
                    self.epoch_reset = True

        # go to next step in sequence
        else:
            # get sentence embeddings
            s1_idx, s2_idx = ep_info['ranking_seq'][t]
            s1 = ep_info['sentence_embs'][s1_idx]
            s2 = ep_info['sentence_embs'][s2_idx]

            # get next state
            state = np.concatenate((
                ep_info['a_emb'],
                ep_info['b_emb'],
                s1,
                s2,
            ))

        return (
            state,
            reward,
            terminated,
            trunc,
            info,
        )
    def _compute_reward(self, ep_info):
        return ranking_reward(
            ep_info,
            self.emb_model,
            K=self.K,
            dev=self.emb_dev,
        )
