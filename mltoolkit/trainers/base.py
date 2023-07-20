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
from mltoolkit.utils import (
    files,
    strings,
    display,
    validate
)
from mltoolkit import cfg_reader

class TrainerBase:
    def __init__(self, config_path, debug=False):
        self.cfg, self.keywords = \
            cfg_reader.load(config_path, debug=debug)
        self.debug = debug

        # set save location for logs
        if debug:
            self.save_loc = f'{files.project_root()}/debug'
            display.debug(f'self.save_loc set to {self.save_loc}')
        elif 'save_loc' in self.cfg.data:
            self.save_loc = self.cfg.data['save_loc']
        else:
            self.save_loc=f'{files.project_root()}/data'

        # set device
        self.dev = self.cfg.model['device']

        # seed the random number generators
        seed = self.cfg.general['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def init_optimizer(self):
        pass

    def init_data(self):
        pass

    def init_model(self):
        pass

    def init_aux(self):
        pass

    def train_step(self, model, batch):
        pass

    def eval_step(self, model, batch):
        pass

    def on_eval_end(self, metric_list, mode):
        pass

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
        loss.backward()

        # gradient clipping
        clip_max_norm = self.cfg.optim['clip_max_norm'] 
        if clip_max_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                clip_max_norm,
                norm_type=self.cfg.optim['clip_norm_type'],
                error_if_nonfinite=False,
                foreach=True
            )
        
        # optimize and reset gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

    def setup(self):

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

        """ dataset section """
        # pre-process dataset 
        display.in_progress('preparing datasets ...')

        self.train_loader, self.val_loader, self.test_loader = \
            self.init_data()

        # check if variables have been defined
        validate.is_assigned(self.train_loader, 'self.train_loader')
        validate.is_assigned(self.val_loader, 'self.val_loader')
        if cfg.data['using_test_loader']:
            validate.is_assigned(
                self.test_loader,
                'self.test_loader',
                extra_info='if you do not intend to assign anything to it, set config -> data -> using_test_loader to False'
            )
        display.done(end='\n\n')

        """ end dataset section """

        """ model section """

        # init model 
        self.model = self.init_model()
        validate.is_assigned(self.model, 'self.model')
        if checkpoint is not None:
            self.model.load_state_dict(
                checkpoint['model'].state_dict()
            )
            display.note('loaded model weights from checkpoint')

        display.in_progress('printing model summary ...')
        print(self.model)

        # count and display learnable parameters
        params = self.model.parameters()
        learnable_params = sum(p.numel() for p in params if p.requires_grad)
        unlearnable_params = sum(p.numel() for p in params if not p.requires_grad)

        display.note(
            f'model has {learnable_params:,} learnable parameters and {unlearnable_params:,} unlearnable parameters'
        )
        display.done(end='\n\n')

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

        """ stochastic weight averaging section """
        display.in_progress('setting up stochastic weight averaging (SWA) ...')

        # define swa model
        self.swa_model = swa_utils.AveragedModel(self.model)

        # define swa scheduler
        anneal_strategy = 'linear' if cfg.optim['swa_strat_is_linear'] else 'cos'
        self.swa_scheduler = swa_utils.SWALR(
            self.optimizer,
            anneal_strategy=anneal_strategy,
            anneal_epochs=cfg.optim['swa_anneal_epochs'],
            swa_lr=cfg.optim['swa_lr'],
        )
        display.note(
            f'SWA scheduler is using the \'{anneal_strategy}\' strategy. '
            + 'you can toggle this option by setting the \'swa_strat_is_linear\' variable to either True or False'
        )

        # set swa beginning
        self.swa_begin = cfg.optim['swa_begin']

        # convert negative swa_begin to actual epoch number
        if self.swa_begin < 0:
            self.swa_begin += cfg.data['num_epochs']
        display.note(
            f'SWA is set to begin at epoch {self.swa_begin}'
        )

        if self.swa_begin < 0 or self.swa_begin >= cfg.data['num_epochs']:
            display.error(
                f'variable \'swa_begin\' is computed to {self.swa_begin} and does not fall into a valid epoch number. ' 
                + f'valid range is: 0 <= swa_begin < {cfg.data["num_epochs"]} for this dataset'
            )
            raise ValueError()

        self.swa_bn_update_steps = cfg.optim['swa_bn_update_steps']
        if self.swa_bn_update_steps == 'all':
            self.swa_bn_update_steps = len(self.train_loader)
        
        # set swa batchnorm update steps
        if self.swa_bn_update_steps < 0 or self.swa_bn_update_steps > len(self.train_loader):
            display.error(
                f'variable \'swa_bn_update_steps\' is set to {self.swa_bn_update_steps} and is not in a valid range. '
                + f'valid range is 0 <= swa_bn_update_steps <= {len(self.train_loader)} for this dataset'
            )
        display.note(
            f'SWA is set to update batchnorm statistics for {self.swa_bn_update_steps} training steps. '
            + 'If your model does not use batchnorm, you can disable this procedure by setting swa_bn_update_steps to 0'
        )

        display.done(end='\n\n')


        """ end stochastic weight averaging section """

        """ auxiliary initialization """
        display.in_progress('initializing auxiliary tools')

        self.init_aux()

        display.done(end='\n\n')

        """ end auxiliary initialization """

        """ tensorboard section """
        # initialize tensorboard logger and log hyperparameters

        display.in_progress('initializing tensorboard logging ...')
        writer = SummaryWriter(
            log_dir=cfg.general['log_dir']
        )
        self._log_hparams(cfg.general["experiment_name"], writer, cfg)
        display.note(
            f'tensorboard writer initialized. access logs at {cfg.general["log_dir"]}'
        )
        display.done(end='\n\n')

        """ end tensorboard section """

        display.title('Finished Setup')

        return writer, ckpt_dir, cfg.general['log_dir'], cfg.general["experiment_name"]

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

    def train(self):

        cfg = self.cfg

        """ setup environment """

        # run through setup of optimizer, model, dataset, tensorboard logger, and checkpoint directory
        (
            writer,
            ckpt_dir,
            log_dir,
            cfg.general["experiment_name"],
        ) = self.setup()

        self.writer = writer

        # initialize checkpointing-related variables
        best_model_score = \
            -math.inf if cfg.model['keep_higher_eval'] else math.inf
        last_ckpt = 0

        # initialize variables related to training progress
        num_epochs = cfg.data['num_epochs']
        steps = 0
        total_steps = len(self.train_loader)*num_epochs
        eval_freq = cfg.data['eval_freq']
        log_freq = cfg.data['log_freq']
        swa_active = False
        swa_step = None

        # initialize variables for formatting and 
        max_epoch_digits = len(str(num_epochs))
        max_step_digits = len(str(total_steps))

        """ end environment setup """

        """ begin training section """

        # initialize progress bar
        display.title('Begin Training')
        prog_bar = tqdm(
            range(total_steps),
            desc=cfg.general["experiment_name"]
        )

        # enter training loop
        for epoch in range(num_epochs):
            for batch in self.train_loader:

                # compute loss and collect metrics
                loss, trn_metrics = self.train_step(self.model, batch)

                # perform optimization step
                self.optim_step(loss)

                # update metrics on progress bar
                prog_bar.set_postfix({
                    'epoch': epoch,
                    'step': steps,
                    'loss': f'{loss.detach().cpu().numpy():.02f}',
                    'swa_active_step': swa_step,
                    'ckpt_step': last_ckpt,
                })

                # log training metrics
                if steps % log_freq == 0:

                    # include defaults in the metrics
                    trn_metrics['scalar'] = trn_metrics.get('scalar', {})
                    trn_metrics['scalar'].update({
                        'loss' : loss,
                        'vars/epoch' : epoch,
                        'vars/lr': \
                            self.scheduler.get_last_lr()[-1]
                            if not swa_active
                            else self.swa_scheduler.get_last_lr()[-1],
                    })

                    # log
                    self._log(writer, trn_metrics, steps, mode='train')

                # log evaluation statistics
                if (steps % eval_freq) == 0:

                    model_score, eval_metrics = self.evaluation_procedure()
                    self._log(writer, eval_metrics, steps, mode='val')

                    # save model state dictionary
                    if self._compare_scores(model_score, best_model_score):
                        torch.save(self.model.state_dict(), f'{ckpt_dir}/best_model.pt')
                        
                        # update trackers
                        last_ckpt = steps
                        best_model_score = model_score

                # update progress bar and increment step counter
                prog_bar.update()
                steps += 1

            """ post-epoch procedure """

            # save checkpoint
            if not swa_active:
                torch.save(
                    {
                        'model': self.model,
                        'optimizer': self.optimizer,
                        'scheduler': self.scheduler,
                        'epoch': epoch
                    },
                    f'{ckpt_dir}/checkpoint.pt'
                )

            # check to use regular scheduler or swa scheduler
            if epoch >= self.swa_begin:

                # set swa flags
                if not swa_active:
                    swa_active = True
                    swa_step = steps

                # perform swa updates
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

            else:
                # apply scheduler step for lr adjustment
                self.scheduler.step()

            """ end post-epoch procedure """

        prog_bar.close()
        display.title('Finished Training')

        """ end training section """
        
        display.title('Begin Final Evaluations')
        """ update batchnorm statistics for swa model """

        if self.swa_bn_update_steps > 0:
            display.in_progress('updating batchnorm statistics on swa model')

            for count, batch in tqdm(
                enumerate(self.train_loader),
                total=self.swa_bn_update_steps,
                desc='updating batchnorm statistics'
            ):

                if count == self.swa_bn_update_steps:
                    break

                self.train_step(self.model, batch)
            print('\r')
            display.done(end='\n\n')

        """ end batchnorm statistics update """

        """ do final evaluation with normal model and then swa model"""

        # do evaluation using normal model
        display.in_progress('evaluating non-swa model')
        model_score, eval_metrics = self.evaluation_procedure(cr=True)

        save_model = False
        if self._compare_scores(model_score, best_model_score):
            torch.save(self.model.state_dict(), f'{ckpt_dir}/best_model.pth')
            best_model_score = model_score

            display.note('non-swa model saved')
        display.done(end='\n\n')

        # do evaluation using swa model
        display.in_progress('evaluating swa model')
        swa_model_score, swa_eval_metrics = self.evaluation_procedure(
            use_swa_model=True,
            cr=True
        )
        if self._compare_scores(swa_model_score, best_model_score):
            # overwrite metrics
            model_score, eval_metrics = swa_model_score, swa_eval_metrics

            # save model
            torch.save(self.swa_model.state_dict(), f'{ckpt_dir}/best_model.pth')
            best_model_score = model_score

            display.note('swa model saved')

        # log last metrics and end 
        self._log(writer, eval_metrics, steps, mode='val')
        display.done(end='\n\n')

        """ end final evaluation """

        """ test set evaluation """
        # evaluate on the test set
        if cfg.data['using_test_loader']:
            display.in_progress('evaluating best model on test set ...')
            model_score, eval_metrics = self.evaluation_procedure(
                use_test_loader=True,
                cr=True
            )
            display.done(end='\n\n')

        """ end test set evaluation """
        
        display.title('End Final Evaluations')

