# external imports
import math
import time
import torch
import numpy as np
import random
import traceback
import os
from copy import deepcopy
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from tqdm import tqdm
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datasets import Dataset

# local imports
from mltoolkit.utils import (
    files,
    strings,
    display,
    validate
)
from mltoolkit import cfg_reader

class Trainer:
    def __init__(self, cfg, debug=False, accelerator=None):

        self.cfg = cfg
        self.debug = debug
        self.experiment_name = cfg.general['experiment_name']
        self.accel = accelerator
        if type(accelerator) != Accelerator:
            self.accel = Accelerator()

        # set save location for logs and checkpointing
        if debug:
            self.results_dir = f'{files.project_root()}/debug/results'
        elif 'results' in self.cfg.paths:
            self.results_dir = self.cfg.paths['results']
            if debug:
                self.results_dir += '/debug'
                display.debug(f'self.results_dir set to {self.results_dir}')
            else:
                self.results_dir += f'/{self.cfg.general["experiment_name"]}'
        else:
            display.error('cfg.paths[\'results\'] not set in config file')
            raise ValueError()

        # seed the random number generators
        seed = self.cfg.general['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _set_var_keys(self, train_vars):
        """
        identifies the dictionary keys in train_vars that identifiy Optimizers, LRSchedulers or Dataloaders
        """

        self.model_keys = []
        self.optim_keys = []
        self.sched_keys = []
        self.loader_keys = []

        for k, v in train_vars.items():
            if issubclass(type(v), Module):
                self.model_keys.append(k)
                continue
            if issubclass(type(v), DataLoader):
                self.loader_keys.append(k)
                continue
            if issubclass(type(v), Optimizer):
                self.optim_keys.append(k)
                continue
            if issubclass(type(v), LRScheduler):
                self.sched_keys.append(k)

    def setup(self):
        display.error("Trainer.setup() not implemented")
        raise NotImplementedError()

    def run_setup(self):
        """
        runs the uder-defined setup function to collect training variabls
        training vars include: models, optimizers, dataloaders, and lr_schedulers
        """
        # collect training vars 
        self.train_vars = self.setup()
        self._set_var_keys(self.train_vars)

        # run training vars through accelerate.prepare and repackage into self.train_vars dict
        prepped_vars = self.accel.prepare(*self.train_vars.values())
        self.train_vars = {k: v for k, v in zip(self.train_vars.keys(), prepped_vars)}

    def train_step(self, batch):
        display.error("Trainer.train_step() not implemented")
        raise NotImplementedError()

    def eval_step(self, batch):
        display.error("Trainer.eval_step() not implemented")
        raise NotImplementedError()

    def on_eval_end(self, metric_list, mode):
        display.error("Trainer.on_eval_end() not implemented")
        raise NotImplementedError()

    def save_criterion(self, cur, prev):
        criterion = self.cfg.params.get('save_criterion', None)
        options = ['max', 'min']

        if criterion == 'max':
            return cur >= prev
        elif criterion == 'min':
            return cur <= prev
        else:
            display.error('cfg.params[save_criterion] not specified choose from: ["max", "min"]')
            raise ValueError()

    def _log(self, writer, metrics, step_number, mode='train'):
        if metrics is None:
            return
        if 'scalar' in metrics:
            for key, val in metrics['scalar'].items():
                writer.add_scalar(
                    f'{key}/{mode}',
                    val,
                    step_number
                )
        if 'scalars' in metrics:
            for key, val in metrics['scalars'].items():
                writer.add_scalars(
                    f'{key}/{mode}',
                    val,
                    step_number
                )
        if 'image' in metrics:
            for key, val in metrics['image'].items():
                writer.add_image(
                    f'{key}/{mode}',
                    val,
                    step_number,
                    dataformats='HWC'
                )
        if 'histogram' in metrics:
            for key, val in metrics['histogram'].items():
                writer.add_histogram(
                    f'{key}/{mode}',
                    val,
                    step_number,
                )
        if 'text' in metrics:
            for key, val in metrics['text'].items():
                writer.add_text(
                    f'{key}/{mode}',
                    val,
                    step_number,
                )

    def _log_hparams(self, name, writer, cfg):
        cfg = cfg._asdict()

        # create markdown table
        table = \
            '---\n# Training Parameters\n---\n\n' + \
            '| parameter | value |\n' + \
            '| --------- | ----- |\n'
        for category, params in cfg.items():
            for key, val in sorted(params.items()):
                table += f'| {category}/**`{key}`** | {str(val)} |\n'

        # write table to tensorboard
        writer.add_text('hparams', table)

    def optim_step(self, loss):
        """
        automates the optimization step for training

        Input:
            loss[Tensor | List[Tensor]]: either a single loss value or list of loss values
        """
        
        is_iterable = True if type(loss) == list else False

        # backward pass and optimizer step, followed by zero_grad()
        try:
            if is_iterable:
                for val in loss:
                    self.accel.backward(val)
            else:
                self.accel.backward(loss)
        except Exception as e:
            display.error(f'Exception occured calling backpropagation at training step: {self.step_counter}')
            traceback.print_exception(e)
            os._exit(1)

        # gradient clipping
        clip_max_norm = self.cfg.params['clip_max_norm'] 
        if clip_max_norm is not None:
            for model_key in self.model_keys:
                nn.utils.clip_grad_norm_(
                    self.train_vars[model_key].parameters(), 
                    clip_max_norm,
                    norm_type=self.cfg.params['clip_norm_type'],
                    error_if_nonfinite=False,
                    foreach=True
                )
        
        # optimize and reset gradients
        for optim in self.optim_keys:
            self.train_vars[optim].step()
            self.train_vars[optim].zero_grad()

    def _setup_tensorboard(self):
        """
        initialize tensorboard logger and log hyperparameters
        """
        cfg = self.cfg

        display.in_progress('initializing tensorboard logging ...')
        writer = SummaryWriter(
            log_dir=cfg.paths['log_dir'] + f'/{self.experiment_name}'
        )
        self._log_hparams(self.experiment_name, writer, cfg)
        display.note(
            f'tensorboard writer initialized. access logs at {cfg.paths["log_dir"]}/{self.experiment_name}'
        )
        display.done(end='\n\n')

        return writer

    def evaluation_procedure(self, use_test_loader=False, cr=False):
        """
        applies no_grad() and model.eval() and then evaluates
        """
        for model in self.model_keys:
            self.train_vars[model].eval()
        with torch.no_grad():

            score, aggregate_metrics = self.evaluate(
                use_test_loader=use_test_loader,
                cr=cr
            )

        for model in self.model_keys:
            self.train_vars[model].train()
        return score, aggregate_metrics

    def evaluate(self, use_test_loader=False, cr=False):
        """
        this function contains the evaluation loop. 
        """

        vloaders = deepcopy(self.loader_keys)
        vloaders.remove('train_loader')

        metric_lists = {name: [] for name in vloaders}

        for loader_key in vloaders:
            if self.accel.is_main_process:
                for batch in tqdm(self.train_vars[loader_key], 
                             total=len(self.train_vars[loader_key]),
                             desc=f'validating on {loader_key}',
                             leave=False):
                    metric_lists[loader_key].append(self.eval_step(batch, loader_key))
            else:
                metric_lists[loader_key] = [
                    self.eval_step(batch, loader_key) 
                    for batch in self.train_vars[loader_key]
                ]
            
        if cr:
            print('\r')

        metric_lists = {
            key: self.accel.gather_for_metrics(metric_lists[key])
            for key in metric_lists.keys()
        }
        score, aggregate_metrics = self.on_eval_end(metric_lists, 'val')
            
        return score, aggregate_metrics

    def train(self, step_limit=None, global_best_score=None, exp_num=None):
        cfg = self.cfg
        save_ckpt = cfg.params['save_checkpoint']

        if self.accel.is_main_process:
            display.in_progress('Running setup() function')
        self.run_setup()
        if self.accel.is_main_process:
            display.done('Finished running setup() function')
        self.accel.wait_for_everyone()

        # initialize directory-related vars
        ckpt_dir = cfg.paths['results'] + f'/{self.experiment_name}'
        if exp_num is not None:
            ckpt_dir = files.dirname(ckpt_dir)
        if self.accel.is_main_process:
            self.writer = self._setup_tensorboard()
            if save_ckpt:
                files.create_path(ckpt_dir)
        self.accel.wait_for_everyone()


        # initialize variables related to training progress
        num_epochs = cfg.params['num_epochs']
        self.step_counter = 0
        total_steps = len(self.train_vars['train_loader'])*num_epochs
        eval_freq = cfg.params['eval_freq']
        log_freq = cfg.params['log_freq']
        skip = cfg.params['skip_first_eval']
        last_ckpt = 0

        # automatically set starting point for best model score based on "save_criterion() function"
        local_best_score = float('inf')
        if self.save_criterion(1, 0):
            local_best_score *= -1
        if global_best_score is None:
            global_best_score = local_best_score

        # initialize variables for formatting and 
        max_epoch_digits = len(str(num_epochs))
        max_step_digits = len(str(total_steps))

        """ begin training section """

        # initialize progress bar
        if self.accel.is_main_process:
            display.title('Begin Training', fill_char='-')
            if step_limit is not None:
                bar_range = min(total_steps, step_limit)
            else:
                bar_range = total_steps
            prog_bar = tqdm(
                range(bar_range),
                desc=cfg.general["experiment_name"]
            )

        # enter training loop
        for epoch in range(num_epochs):
            for batch in self.train_vars['train_loader']:

                # compute loss and collect metrics
                loss, trn_metrics = self.train_step(batch)

                # perform optimization step
                self.optim_step(loss)

                if self.accel.is_main_process:

                    # update metrics on progress bar
                    prog_bar.set_postfix({
                        'epoch': epoch,
                        'step': self.step_counter,
                        'loss': f'{float(loss.detach().cpu()):.02f}',
                        'ckpt_step': last_ckpt,
                    })

                    # log training metrics
                    if self.step_counter % log_freq == 0:
                        # set loss values for train metrics dict
                        trn_metrics['scalar'] = trn_metrics.get('scalar', {})
                        if type(loss) == list:
                            trn_metrics['scalar'].update({f'loss{i:02}': l for i, l in enumerate(loss)})
                        else:
                            trn_metrics['scalar'].update({'loss': loss})

                        # set learning rate values for train metrics dict
                        trn_metrics['scalar'].update({
                            f'vars/lr-{sk}': self.train_vars[sk].get_last_lr()[-1] 
                            for sk in self.sched_keys
                        })

                        # include defaults in the metrics
                        trn_metrics['scalar'].update({
                            'vars/epoch' : epoch,
                        })

                        self._log(self.writer, trn_metrics, self.step_counter, mode='train')

                # log evaluation statistics
                if (self.step_counter % eval_freq) == 0 and not skip:

                    last_score, eval_metrics = self.evaluation_procedure()

                    if self.accel.is_main_process:
                        self._log(self.writer, eval_metrics, self.step_counter, mode='val')

                    # save model state dictionary
                    if self.save_criterion(last_score, local_best_score):

                        local_best_score = last_score
                        if save_ckpt and self.save_criterion(local_best_score, global_best_score):

                            # update trackers
                            last_ckpt = self.step_counter
                            global_best_score = local_best_score
                            for model in self.model_keys:
                                self.accel.wait_for_everyone()
                                self.accel.save_model(
                                    self.train_vars[model],
                                    f'{ckpt_dir}/{model}-best_model',
                                    max_shard_size="5GB",
                                    safe_serialization=True
                                )
                    else:
                        print('skipped saving')

                if step_limit is not None and step_limit == self.step_counter:
                    if self.accel.is_main_process:
                        display.done(f'Step limit reached. best model score: {local_best_score}')
                    self.accel.wait_for_everyone()
                    return last_score

                # set skip value so it can enter back into the evaluation procedure
                skip = False

                # update progress bar and increment step counter
                if self.accel.is_main_process:
                    prog_bar.update()
                self.step_counter += 1

            for sched in self.sched_keys:
                self.train_vars[sched].step()

        if self.accel.is_main_process:
            prog_bar.close()
            display.title('Finished Training', fill_char='-')

        """ end training section """

        return last_score
       
