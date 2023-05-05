# external imports
import torch
import numpy as np
import datasets
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
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
from mltoolkit import cfg_reader
class TrainerBase:
    def __init__(self, config_path, debug=False):
        self.debug = debug
        self.debug_dir = f'{files.project_root()}/debug'
        self.model = None
        self.cfg, self.keywords = \
            cfg_reader.load(config_path, debug=debug)

        # set device
        self.dev = self.cfg.model['device']

        # seed the random number generators
        torch.manual_seed(self.cfg.general['seed'])
        np.random.seed(self.cfg.general['seed'])
        self.rng = np.random.default_rng(
            self.cfg.general['seed']
        )

    def init_optimizer(self):
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.optim['lr'],
            weight_decay=self.cfg.optim['weight_decay']
        )

    def init_loss_fn(self):
        pass

    def prepare_data_and_tools(self):
        pass

    def evaluate(self):
        return 0, {}

    def test(self):
        pass

    def train_step(self, batch):
        return torch.tensor(0), {}

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

        # create checkpoint directory
        ckpt_dir = cfg.model['ckpt_dir']
        files.create_path(ckpt_dir)
        print(strings.green(
            f'\ncheckpoints set to be saved to {ckpt_dir}'
        ))

        # pre-process dataset 
        print(strings.green(
            '\nPreparing dataset and tools...'
        ))
        self.prepare_data_and_tools()

        # initialize tensorboard logger and log hyperparameters
        writer = SummaryWriter(
            log_dir=cfg.general['log_dir']
        )

        self._log_hparams(cfg.general["experiment_name"], writer, cfg)
        print(strings.green(
            f'\ntensorboard initialized. access logs at {cfg.general["log_dir"]}'
        ))

        return writer, ckpt_dir, cfg.general['log_dir'], cfg.general["experiment_name"]

    def train(self):
        cfg = self.cfg

        # initialize for training
        (writer, ckpt_dir,
         log_dir, cfg.general["experiment_name"]) = self.setup()

        # use these for checkpointing
        best_model_score = -math.inf
        best_model_step = -1

        # prepare for training
        num_epochs = cfg.data['num_epochs']
        steps = 0
        total_steps = len(self.ds['train'])*num_epochs
        display.title('Begin Training')
        prog_bar = tqdm(
            range(total_steps),
            desc=cfg.general["experiment_name"]
        )

        # break dataset into shards
        num_shards = cfg.data['num_shards']
        shard_ids = np.arange(num_shards)
        shuffle = cfg.data['shuffle']
        batch_size = cfg.data['batch_size']
        eval_freq = cfg.data['eval_freq']
        log_freq = cfg.data['log_freq']

        if shuffle:
            np.random.shuffle(shard_ids)

        for epoch, shard in product(range(num_epochs), shard_ids):

            # split get shard and shuffle if specified
            ds_shard = self.ds['train'].shard(
                num_shards=num_shards,
                index=shard
            )
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

                
                trn_metrics.update({
                    'scalar' : {
                        'loss/train' : loss.detach().cpu().numpy(),
                        'epoch' : epoch
                    }
                })
                # log training statistics
                if steps % log_freq == 0 or steps == total_steps-1:
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

        self.test()
        display.title('Finished Training')

