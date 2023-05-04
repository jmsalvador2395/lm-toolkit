# external imports
import torch
import numpy as np
import datasets
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from random import randint
import math
from itertools import product
import time

# local imports
from mltoolkit.utils import (
    files,
    strings,
    display,
    data
)
class TrainerBase:
    def __init__(self, cfg, keywords):
        self.model = None
        self.cfg = cfg
        self.keywords = keywords

        # set device
        self.dev = cfg.model.get('device', 'cpu')

        # seed the random number generators
        torch.manual_seed(
            cfg.general.get(
                'seed',
                randint(0, 2**32)
            )
        )
        np.random.seed(
            cfg.general.get(
                'seed', 
                randint(0, 2**32)
            )
        )
        self.rng = np.random.default_rng(
            seed=cfg.general.get(
                'seed', 
                randint(0, 2**32)
            )
        )

    def init_optimizer(self):
        pass

    def init_loss_fn(self):
        pass

    def prepare_data(self):
        pass

    def evaluate(self):
        return 0, {}

    def train_step(self, batch):
        return torch.randn(1)[0]

    def _log(writer, metrics, step_number):
        if metrics is None:
            return
        if 'scalar' in metrics:
            for metric in metrics['scalar'].keys():
                writer.add_scalar(
                    metric,
                    metrics['scalar'][metric],
                    step_number
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
                table += f'| {category}/**`{key}`** | {val} |\n'

        # write table to tensorboard
        writer.add_text('hparams', table)

    def _log(self, writer, metrics, step_number):
        if metrics is None:
            return
        if 'scalar' in metrics:
            for metric in metrics['scalar'].keys():
                writer.add_scalar(
                    metric,
                    metrics['scalar'][metric],
                    step_number
                )

    def optim_step(self, loss):
        # backward pass and optimizer step, followed by zero_grad()
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

    def setup(self):
        cfg = self.cfg

        self.init_optimizer()
        self.init_loss_fn()

        # set experiment name
        experiment_name = f'{self.keywords["timestamp"]}-{cfg.model["name"]}'

        # create checkpoint directory
        ckpt_dir = (
            cfg.model.get(
                'ckpt_dir', 
                f'files.get_project_root()/checkpoints'
            ).rstrip('/') + \
            f'/{experiment_name}'
        )
        files.create_path(ckpt_dir)
        print(strings.green(
            f'\ncheckpoints set to be saved to {ckpt_dir}'
        ))

        # pre-process dataset 
        print(strings.green('\nPreparing dataset for training...'))
        self.prepare_data()

        # initialize tensorboard logger and log hyperparameters
        log_dir = cfg.general.get(
            'logdir_base',
            f'{files.get_project_root()}/tensorboard'
        ).rstrip('/') + \
        f'/{experiment_name}'
        cfg.general.update({
            'log_dir' : log_dir
        })
        cfg.general.pop('logdir_base')
        writer = SummaryWriter(
            log_dir=log_dir
        )
        self._log_hparams(experiment_name, writer, cfg)
        print(strings.green('\nlogging to {log_dir}'))

        return writer, ckpt_dir, log_dir, experiment_name

    def train(self):
        cfg = self.cfg

        # initialize for training
        (writer, ckpt_dir,
         log_dir, experiment_name) = self.setup()

        # use these for checkpointing
        best_model_score = -math.inf
        best_model_step = -1

        # prepare for training
        num_epochs = cfg.data.get('num_epochs', 1)
        steps = 0
        total_steps = len(self.ds['train'])*num_epochs
        display.title('Begin Training')
        prog_bar = tqdm(
            range(total_steps),
            desc=experiment_name
        )

        # break dataset into shards
        num_shards = cfg.data.get('num_shards', 1)
        shard_ids = np.arange(num_shards)
        shuffle = cfg.data.get('shuffle', True)
        batch_size = cfg.data.get('batch_size', 32)
        eval_freq = cfg.data.get('eval_freq', 1000)
        log_freq = cfg.data.get('log_freq', 1000)
        if shuffle:
            np.random.shuffle(shard_ids)

        for epoch, shard in product(range(num_epochs), shard_ids):

            # split get shard and shuffle if specified
            ds_shard = self.ds['train'].shard(num_shards=num_shards, index=shard)
            if shuffle:
                ds_shard = ds_shard.shuffle(generator=self.rng)

            #for batch in DataLoader(ds_shard, batch_size=batch_size, shuffle=shuffle):
            for i in range(0, len(ds_shard), batch_size):
                batch = ds_shard[i:i+batch_size]
               
                loss, trn_metrics = self.train_step(batch)
                self.optim_step(loss)

                # update tracked parameters
                prog_bar.set_postfix({
                    'loss' : f'{loss.detach().cpu().numpy():.02f}',
                    'epoch' : epoch,
                    'step' : steps,
                })

                
                # log training statistics
                if steps % log_freq == 0 or steps == total_steps-1:
                    trn_metrics = {
                        'scalar' : {
                            'loss/train' : loss.detach().cpu().numpy(),
                            'epoch' : epoch
                        }
                    }
                    self._log(writer, trn_metrics, steps)

                # log evaluation statistics
                if steps % eval_freq == 0 or steps == total_steps-1:
                    model_score, eval_metrics = self.evaluate()
                    self._log(writer, eval_metrics, steps)

                    if model_score > best_model_score:
                        torch.save(self.model, f'{ckpt_dir}/best_model.pth')
                        print(strings.green(
                            f'best model saved at step {steps}'
                        ))
                        best_model_step = steps
                        best_model_score = model_score


                # update progress bar and counter
                prog_bar.update(batch_size)
                steps += 1

        prog_bar.close()
        display.title('Finished Training')

