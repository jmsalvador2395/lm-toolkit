# external imports
import math
import time
import torch
import numpy as np
import random
import gymnasium as gym
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import swa_utils
from itertools import product
from datasets import Dataset
from copy import deepcopy

# for typing
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Tuple, List, Dict, TypeVar, Any

# local imports
from mltoolkit.trainers.gym_base import TrainerBaseGym
from mltoolkit.utils import (
    files,
    strings,
    display,
    validate
)
from mltoolkit import cfg_reader
from mltoolkit.gym_environments import AllSidesRankingEnv
from mltoolkit.models import ActorCriticMLP

class TrainerAllSidesRanking(TrainerBaseGym):

    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug=debug)
    
    def __init__(self, config_path, debug=False):
        """
        initialization function. ideally there is no need to edit this since initialization is performed at designated functions

        Input:
            config_path[str]: the location of the YAML config. parent class takes care of loading the config.
            debug[bool]: used to set debug-related parameters. affected parameters will be printed to console
        """
        super().__init__(config_path, debug)

    def init_env(self) -> Tuple[gym.Env, Dict]:
        """
        This function is used to intialize your gym environment (Farama gymnasium)

        Return:
            env[gym.Env]: the RL environment
            reset_options[Dict]: the dictionary passed to the initial reset call of the environment
        """
        env = AllSidesRankingEnv(self.cfg)

        options = {
            'full_reset': True,
        }
        #env.reset()

        return (env, options)

    def init_model(self):
        """
        use this function to initialize your model.
        feel free to initialize any other models here and just assign them as self.<other model>

        Return
            model[nn.Module]: the model used for training
        """

        cfg = self.cfg
        self.model = ActorCriticMLP(cfg).to(self.dev)

    def _freeze_params(self, model):
        
        # freeze the parameters
        for param in model.parameters():
            param.requires_grad = False

        return model

    def init_optimizer(self) -> Tuple[Optimizer, LRScheduler]:
        """
        initialize your optimizer and learning rate schedulers here

        Return
            optim_tools[Tuple[Optimizer, LRScheduler]: a tuple that includes the optimizer and scheduler
        """

        cfg = self.cfg

        # optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optim['lr'],
            betas=(
                cfg.optim.get('beta1', .9),
                cfg.optim.get('beta2', .999)
            ),
            eps=float(cfg.optim.get('eps', 1e-8)),
            weight_decay=float(self.cfg.optim.get('weight_decay', 1e-4)),
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.optim['sched_step_size'],
            gamma=cfg.optim['sched_gamma'],
        )
        
        return (
            optimizer,
            scheduler
        )

    def init_aux(self):
        """
        Use this function to initialize any other important variables that you want to use for training.
        """
        self.loss_fn = nn.CrossEntropyLoss()
        self._reset_trajectory()
        self.result = None
        self.epoch_end = False
        self.ep_count = 0

    def action_step(self, state: torch.Tensor) -> int:
        """
        chooses an action based on the current state. get valid actions by calling 'self.env.action_space'

        Input
            s[torch.Tensor]: the input state of shape (3072,)

        Output
            a[int]: the chosen action
        """

        # compute action and append to history
        self.model.eval()
        with torch.no_grad():
            
            state = torch.tensor(
                state, 
                dtype=torch.float32,
                device=self.dev
            )
            dist, value = self.model(state)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            self.trajectory['states'].append(state)
            self.trajectory['actions'].append(action)
            self.trajectory['log_probs'].append(log_prob)
            self.trajectory['values'].append(value)

        self.model.train()
        
        return action.detach().cpu().numpy()

    def on_step_end(self, next_state, reward, terminated, trunc, info):

        # store reward information to history
        self.trajectory['rewards'].append(reward)
        if terminated:
            info.pop('end_of_epoch')
            self.result = info

    def on_episode_end(self):

        self.ep_count += 1

        if self.ep_count % 2 != 0:
            self.result = None
            return

        ############### prepare variables ############### 
        
        cfg = self.cfg

        gamma = cfg.optim['discount']
        T = len(self.trajectory['states'])
        batch_size = cfg.data['batch_size']
        ppo_epochs = cfg.data['num_ppo_epochs']
        epsilon = cfg.optim['clip_epsilon']
        c1 = cfg.optim['c1']
        c2 = cfg.optim['c2']

        metrics = {
            'loss': [],
        }

        #################################################

        ############ prepare tensors from episode history ############ 

        values = torch.cat(self.trajectory['values'])
        rewards = torch.tensor(self.trajectory['rewards'], device=self.dev)
        states = torch.stack(self.trajectory['states'])
        actions = torch.stack(self.trajectory['actions'])
        log_probs = torch.stack(self.trajectory['log_probs'])
    
        ##############################################################

        steps = torch.arange(len(states))
        returns = torch.tensor(
            [
                torch.sum(gamma**torch.arange(T-t, device=self.dev)*rewards[t:])
                for t in steps
            ],
            device=self.dev
        )
        advantages = returns - values

        # do PPO updates
        indices = np.arange(T)

        # do multiple iterations for each episode
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            shuffled_indices = [
                indices[i:i+batch_size] 
                for i in range(0, T, batch_size)
            ]
            
            # loop for each minibatch
            for batch_ids in shuffled_indices:

                # get batch from batch_ids
                action_batch = actions[batch_ids]
                state_batch = states[batch_ids]
                return_batch = returns[batch_ids]
                log_prob_batch = log_probs[batch_ids]
                advantage_batch = advantages[batch_ids]

                # compute ratio numerator
                dist, value = self.model(state_batch)
                value = torch.flatten(value)

                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action_batch)
                ratio = (new_log_probs - log_prob_batch).exp()

                surr1 = ratio*advantage_batch
                surr2 = \
                    torch.clamp(ratio, 1 - epsilon, 1 + epsilon) \
                    * advantage_batch

                actor_loss = -torch.mean(torch.min(surr1, surr2))
                critic_loss = torch.mean((return_batch - value)**2)

                loss = actor_loss + c1*critic_loss - c2*entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                metrics['loss'].append(loss.detach().cpu().numpy())


        # average the metrics
        metrics['loss'] = np.mean(metrics['loss'])
        metrics['cumulative reward'] = torch.sum(rewards).cpu().numpy()
        metrics['last reward'] = rewards[-1].detach().cpu().numpy()

        self._log(
            self.writer,
            {
                'scalar': metrics,
                'histogram': {
                    'action distribution': actions
                }
            },
            self.global_ep_count
        )

        display.debug(f'{self.cfg.general["experiment_name"]} rewards: {rewards[rewards != 0]}')
        display.debug(f'spearman score: {self.result["spearman"]}')
        display.debug(f'ndcg score: {self.result["ndcg"]}')
        display.debug(f'len a: {self.env.ep_info["len_a"]}')
        display.debug(f'len b: {self.env.ep_info["len_b"]}')
        display.debug(f'len: {self.env.ep_info["len_a"] + self.env.ep_info["len_b"]}')
        display.debug(f'rankings: {self.env.ep_info["rankings"]}')
        display.note(f'**PREDICTION**: {self.result["top_doc"]}')
        display.note(f'**TRUE INTERSECTION**: {self.result["intersection"]}', end='\n\n')

        # reset trajectories
        self._reset_trajectory()
        self.result = None

    def _reset_trajectory(self):

        self.trajectory = {
            'rewards': [],
            'actions': [],
            'states': [],
            'values': [],
            'log_probs': [],
        }

    def build_reset_options(self):
        return {}

