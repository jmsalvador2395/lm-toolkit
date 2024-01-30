# external imports
import math
import time
import torch
import numpy as np
import random
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import swa_utils
from itertools import product
from datasets import Dataset

# local imports
from mltoolkit.trainers.base import TrainerBase
from mltoolkit.utils import (
    files,
    strings,
    display,
    validate
)

class TrainerBaseGym(TrainerBase):

    def __init__(self, config_path, debug=False):
        super().__init__(config_path, debug=debug)

    def setup(self):

        cfg = self.cfg
        display.title('Setting Up Environment')

        # initialize primary variables. these will be assigned as we run through the setup process
        self.env = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        checkpoint = None # this will be used for checkpointing logic

        """ checkpointing section """

        display.in_progress('setting up checkpoint directory ...')

        # create checkpoint save directory
        ckpt_dir = cfg.model['ckpt_dir']
        files.create_path(ckpt_dir)

        if self.debug:
            display.debug(f'checkpoints set to be saved to {ckpt_dir}')
        else:
            display.note(f'checkpoints set to be saved to {ckpt_dir}')
        display.done(end='\n\n')

        # load chackpoint if specified
        if cfg.general['load_checkpoint'] is not None:
            validate.path_exists(cfg.general['load_checkpoint'])
            display.in_progress(f'loading checkpoint from: {cfg.general["load_checkpoint"]}')
            
            checkpoint = torch.load(cfg.general['load_checkpoint'])

            display.done('\n\n')

        """ end checkpointing section """

        """ gym environment section """

        # load the environment
        display.in_progress('preparing environment...')
        self.env, env_reset_options = self.init_env()

        # check if variables have been defined
        validate.is_assigned(self.env, 'self.env')
        display.done(end='\n\n')

        """ end environment section """

        """ model section """

        # init model 
        self.init_model()

        """ end model section """

        """ optimizer section """

        # initialize optimizer and throw errors if self.optim is not initialized
        display.in_progress('initializing optimizer and lr scheduler ...')
        self.optimizer, self.scheduler = self.init_optimizer()
        if checkpoint is not None:
            self.optimizer.load_state_dict(
                checkpoint['optimizer'].state_dict()
            )
            self.scheduler.load_state_dict(
                checkpoint['scheduler'].state_dict()
            )

        validate.is_assigned(self.optimizer, 'self.optimizer')
        validate.is_assigned(self.scheduler, 'self.scheduler')
        display.done(end='\n\n')

        """ end optimizer section """

        """ auxiliary initialization """
        display.in_progress('initializing auxiliary tools')

        self.init_aux()

        display.done(end='\n\n')

        """ end auxiliary initialization """

        """ tensorboard section """
        # initialize tensorboard logger and log hyperparameters

        display.in_progress('initializing tensorboard logging ...')
        self.writer = SummaryWriter(
            log_dir=cfg.general['log_dir']
        )
        self._log_hparams(cfg.general["experiment_name"], self.writer, cfg)
        display.note(
            f'tensorboard writer initialized. access logs at {cfg.general["log_dir"]}'
        )
        display.done(end='\n\n')

        """ end tensorboard section """

        display.title('Finished Setup')

        return (
            ckpt_dir,
            cfg.general['log_dir'],
            cfg.general["experiment_name"],
            env_reset_options,
        )

    def _compare_scores(self, model_score, best_model_score):
        """
        compares evaluation scores and makes decision to checkpoint based based on user config
        """
        if self.cfg.model['keep_higher_eval']:
            return model_score >= best_model_score
        else:
            return model_score <= best_model_score

    def evaluation_procedure(self, use_test_loader=False, use_swa_model=False, cr=False):
        """
        applies no_grad() and model.eval() and then evaluates
        """
        with torch.no_grad():
            self.model.eval()
            self.swa_model.eval()

            score, aggregate_metrics = self.evaluate(
                use_test_loader=use_test_loader,
                use_swa_model=use_swa_model,
                cr=cr
            )

            self.model.train()
            self.swa_model.train()
        return score, aggregate_metrics

    def evaluate(self, use_test_loader=False, use_swa_model=False, cr=False):
        """
        this function contains the evaluation loop
        """

        # set model
        model = self.swa_model if use_swa_model else self.model

        # set mode
        mode = 'test' if use_test_loader else 'val'

        metric_list = []
        dl = self.test_loader if use_test_loader else self.val_loader

        for batch in tqdm(dl, total=len(dl), desc='validating', leave=False):
            metric_list.append(self.eval_step(model, batch, mode))
        if cr:
            print('\r')

        metric_list = Dataset.from_list(metric_list)
        score, aggregate_metrics = self.on_eval_end(metric_list, mode)
            
        return score, aggregate_metrics

    def on_step_end(self, s, r, terminated, trunc, info):
        display.error('need to implement after_step() function')
        raise NotImplementedError()

    def on_episode_end(self):
        pass

    def build_reset_options(self):
        display.error('need to implement build_reset_options() function')
        raise NotImplementedError()

    def train(self):

        ################ set training variables ################

        cfg = self.cfg
        num_episodes = cfg.data['num_episodes']

        ########################################################

        """ setup environment """

        # run through setup of optimizer, model, dataset, tensorboard logger, and checkpoint directory
        (
            ckpt_dir,
            log_dir,
            cfg.general["experiment_name"],
            env_reset_options,
        ) = self.setup()

        self.global_ep_count = 0

        prev_state, info = self.env.reset(
            seed=cfg.general['seed'], 
            options=env_reset_options
        )
        self.env.render()

        for ep in range(num_episodes):

            terminated = False
            while not terminated:

                # choose action
                action = self.action_step(prev_state)

                # make step and then collect results
                state, reward, terminated, trunc, info = self.env.step(action)
                self.env.render()

                # run through post-step procedure
                self.on_step_end(state, reward, terminated, trunc, info)

                # assign prev_state for next iteration
                prev_state = state
                
                # break on episode end
                if terminated: break

            # run through post-episode procedure
            self.on_episode_end()

            # reset environment
            options = self.build_reset_options()
            self.env.reset(options=options)

            # increment episode counter
            self.global_ep_count += 1


