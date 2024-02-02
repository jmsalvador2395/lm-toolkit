# external imports
import math
import time
import torch
import numpy as np
import random
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from itertools import product
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
    def __init__(self, cfg, debug=False):
        self.cfg = cfg
        self.debug = debug

        # initialize accelerator
        self.accelerator = Accelerator()

        # set save location for logs
        if debug:
            self.save_loc = f'{files.project_root()}/debug'
            display.debug(f'self.save_loc set to {self.save_loc}')
        elif 'save_loc' in self.cfg.paths:
            self.save_loc = self.cfg.paths['save_loc'] + f'/{self.cfg.general["experiment_name"]}'
        else:
            self.save_loc=f'{files.project_root()}/data/{self.cfg.general["experiment_name"]}'

        # set deepspeed batch size
        try:
            AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = \
                cfg.params['batch_size']

            display.info('set accelerator state')
        except:
            pass

        # seed the random number generators
        seed = self.cfg.general['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        (self.model,
         self.train_loader, 
         self.val_loader,
         self.optimizer,
         self.scheduler) = self.setup()

        self.model, self.train_loader, self.val_loader, self.optimizer, self.scheduler = \
            self.accelerator.prepare(
                self.model,
                self.train_loader, 
                self.val_loader,
                self.optimizer,
                self.scheduler
         )

    def setup(self):
        display.error("Trainer.setup() not implemented")
        raise NotImplementedError()

    def train_step(self, batch):
        display.error("Trainer.train_step() not implemented")
        raise NotImplementedError()

    def eval_step(self, batch):
        display.error("Trainer.eval_step() not implemented")
        raise NotImplementedError()

    def on_eval_end(self, metric_list, mode):
        display.error("Trainer.on_eval_end() not implemented")
        raise NotImplementedError()

    def save_criterion(self, new_score, prev_best):
        display.error("Trainer.save_criterion() not implemented")
        raise NotImplementedError()

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

        # backward pass and optimizer step, followed by zero_grad()
        #loss.backward()
        self.accelerator.backward(loss)

        # gradient clipping
        clip_max_norm = self.cfg.params['clip_max_norm'] 
        if clip_max_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                clip_max_norm,
                norm_type=self.cfg.params['clip_norm_type'],
                error_if_nonfinite=False,
                foreach=True
            )
        
        # optimize and reset gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _setup_tensorboard(self):
        """
        initialize tensorboard logger and log hyperparameters
        """
        cfg = self.cfg

        display.in_progress('initializing tensorboard logging ...')
        writer = SummaryWriter(
            log_dir=cfg.paths['log_dir']
        )
        self._log_hparams(cfg.general["experiment_name"], writer, cfg)
        display.note(
            f'tensorboard writer initialized. access logs at {cfg.paths["log_dir"]}'
        )
        display.done(end='\n\n')

        return writer

    def _setup_legacy(self):
        """
        old setup function
        """

        cfg = self.cfg
        display.title('Setting Up Environment')

        # initialize primary variables. these will be assigned as we run through the setup process
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
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

        display.title('Finished Setup')

        return writer, ckpt_dir, cfg.general['log_dir'], cfg.general["experiment_name"]

    def evaluation_procedure(self, use_test_loader=False, cr=False):
        """
        applies no_grad() and model.eval() and then evaluates
        """
        with torch.no_grad():
            self.model.eval()

            score, aggregate_metrics = self.evaluate(
                use_test_loader=use_test_loader,
                cr=cr
            )

            self.model.train()
        return score, aggregate_metrics

    def evaluate(self, use_test_loader=False, cr=False):
        """
        this function contains the evaluation loop. 
        """

        metric_list = []

        if self.accelerator.is_main_process:
            for batch in tqdm(self.val_loader, total=len(self.val_loader), desc='validating', leave=False):
                metric_list.append(self.eval_step(batch, 'val'))
        else:
            metric_list = [self.eval_step(batch, 'val') for batch in self.val_loader]
            
        if cr:
            print('\r')

        metric_list = self.accelerator.gather_for_metrics(metric_list)
        score, aggregate_metrics = self.on_eval_end(metric_list, 'val')
            
        return score, aggregate_metrics

    def train(self):
        cfg = self.cfg

        # initialize directory-related vars
        if self.accelerator.is_main_process:
            self.writer = self._setup_tensorboard()
            ckpt_dir = cfg.paths['ckpt_dir']
            files.create_path(ckpt_dir)


        # initialize variables related to training progress
        num_epochs = cfg.params['num_epochs']
        self.step_counter = 0
        total_steps = len(self.train_loader)*num_epochs
        eval_freq = cfg.params['eval_freq']
        log_freq = cfg.params['log_freq']
        last_ckpt = 0

        # automatically set starting point for best model score based on "save_criterion() function"
        best_model_score = float('inf')
        if self.save_criterion(1, 0):
            best_model_score *= -1

        # initialize variables for formatting and 
        max_epoch_digits = len(str(num_epochs))
        max_step_digits = len(str(total_steps))

        """ begin training section """

        # initialize progress bar
        if self.accelerator.is_main_process:
            display.title('Begin Training')
            prog_bar = tqdm(
                range(total_steps),
                desc=cfg.general["experiment_name"]
            )

        # enter training loop
        for epoch in range(num_epochs):
            for batch in self.train_loader:

                # compute loss and collect metrics
                loss, trn_metrics = self.train_step(batch)

                # perform optimization step
                self.optim_step(loss)

                if self.accelerator.is_main_process:

                    # update metrics on progress bar
                    prog_bar.set_postfix({
                        'epoch': epoch,
                        'step': self.step_counter,
                        'loss': f'{float(loss.detach().cpu()):.02f}',
                        'ckpt_step': last_ckpt,
                    })

                    # log training metrics
                    if self.step_counter % log_freq == 0:

                        # include defaults in the metrics
                        trn_metrics['scalar'] = trn_metrics.get('scalar', {})
                        trn_metrics['scalar'].update({
                            'loss' : loss,
                            'vars/epoch' : epoch,
                            'vars/lr': self.scheduler.get_last_lr()[-1]
                        })

                        self._log(self.writer, trn_metrics, self.step_counter, mode='train')

                # log evaluation statistics
                if (self.step_counter % eval_freq) == 0:

                    model_score, eval_metrics = self.evaluation_procedure()

                    if self.accelerator.is_main_process:
                        self._log(self.writer, eval_metrics, self.step_counter, mode='val')

                        # save model state dictionary
                        if self.save_criterion(model_score, best_model_score):
                            torch.save(self.model.state_dict(), f'{ckpt_dir}/best_model.pt')
                            
                            # update trackers
                            last_ckpt = self.step_counter
                            best_model_score = model_score

                # update progress bar and increment step counter
                if self.accelerator.is_main_process:
                    prog_bar.update()
                self.step_counter += 1

                self.scheduler.step()

        if self.accelerator.is_main_process:
            prog_bar.close()
            display.title('Finished Training')

        """ end training section """
       
