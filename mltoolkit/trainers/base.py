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
    data,
    validate
)
from mltoolkit import cfg_reader
class TrainerBase:
    def __init__(self, config_path, debug=False):
        self.cfg, self.keywords = \
            cfg_reader.load(config_path, debug=debug)
        self.debug = debug

        if debug:
            self.save_loc = f'{files.project_root()}/debug'
        elif 'save_loc' in self.cfg.data:
            self.save_loc = self.cfg.data['save_loc']
        else:
            self.save_loc=f'{files.project_root()}/data'
        self.model = None

        # set device
        self.dev = self.cfg.model['device']

        # seed the random number generators
        torch.manual_seed(self.cfg.general['seed'])
        np.random.seed(self.cfg.general['seed'])
        self.rng = np.random.default_rng(
            self.cfg.general['seed']
        )
        self.scheduler = None

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

    def init_model(self):
        self.model = torch.nn.Linear(2, 2)

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
                table += f'| {category}/**`{key}`** | {str(val)} |\n'

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
        self.optim.step()

        self.optim.zero_grad()

    def setup(self):
        cfg = self.cfg

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

        # init model and optimizer
        if cfg.model['load_checkpoint'] is None:
            self.init_model()
        else:
            validate.path_exists(
                cfg.model['load_checkpoint'],
                extra_info=f'given checkpoint path \'{cfg.model["load_checkpoint"]}\' is invalid'
            )
            print(strings.green(f'loading checkpoint from: {cfg.model["load_checkpoint"]}'))
            self.model = torch.load(cfg.model['load_checkpoint'])

        print(strings.green('\nprinting model summary ...'))
        print(self.model)
        params = self.model.parameters()
        num_params = sum(p.numel() for p in params if p.requires_grad)
        print(strings.green(
            f'\nmodel has {num_params:,} learnable parameters'
        ))
        self.init_optimizer()

        # initialize tensorboard logger and log hyperparameters
        writer = SummaryWriter(
            log_dir=cfg.general['log_dir']
        )

        self._log_hparams(cfg.general["experiment_name"], writer, cfg)
        print(strings.green(
            f'\ntensorboard initialized. access logs at {cfg.general["log_dir"]}'
        ))

        return writer, ckpt_dir, cfg.general['log_dir'], cfg.general["experiment_name"]

    def _compare_model_scores(self, model_score, best_model_score):
        if self.cfg.model['keep_higher_eval']:
            return model_score >= best_model_score
        else:
            return model_score <= best_model_score

    def train(self):
        cfg = self.cfg

        # initialize for training
        (writer, ckpt_dir,
         log_dir, cfg.general["experiment_name"]) = self.setup()

        # use these for checkpointing
        best_model_score = -math.inf if cfg.model['keep_higher_eval'] \
                           else math.inf
        last_ckpt = 0

        # prepare for training
        num_epochs = cfg.data['num_epochs']
        steps = 0
        total_steps = len(self.ds['train'])*num_epochs
        max_epoch_digits = len(str(num_epochs))
        max_step_digits = len(str(total_steps))
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

        # make tracker to track the last checkpoint step
        for epoch, shard in product(range(num_epochs), shard_ids):

            # split get shard and shuffle if specified
            ds_shard = self.ds['train'].shard(
                num_shards=num_shards,
                index=shard
            )
            if shuffle:
                ds_shard = ds_shard.shuffle(
                    generator=self.rng,
                    keep_in_memory=True,
                    load_from_cache_file=False,
                )

            #for batch in DataLoader(ds_shard, batch_size=batch_size, shuffle=shuffle):
            progress=0
            for i in range(0, len(ds_shard), batch_size):
                batch = ds_shard[i:i+batch_size]
                update_step = \
                    batch_size if i+batch_size < len(ds_shard) \
                    else len(ds_shard) - i
                progress += update_step
               
                loss, trn_metrics = self.train_step(batch)
                self.optim_step(loss)

                # update tracked parameters
                prog_bar.set_postfix({
                    'loss' : f'{loss.detach().cpu().numpy():.02f}',
                    'epoch' : epoch,
                    'step' : steps,
                    'last_ckpt': last_ckpt
                })

                trn_metrics.get('scalar', {}).update({
                    'loss/train' : loss,
                    'epoch' : epoch
                })
                # log training statistics
                if steps % log_freq == 0 or steps == total_steps-1:
                    self._log(writer, trn_metrics, steps)

                # log evaluation statistics
                if (steps % eval_freq == 0 or steps == total_steps-1):
                    with torch.no_grad():
                        if cfg.model['evaluate']:
                            self.model.eval()
                            model_score, eval_metrics = self.evaluate()
                            self.model.train()
                        else:
                            model_score, eval_metrics = 0, {}
                    self._log(writer, eval_metrics, steps)

                    if self._compare_model_scores(model_score, best_model_score) \
                    and self.cfg.model['save_checkpoint']:
                        torch.save(self.model, f'{ckpt_dir}/best_model.pth')
                        
                        # update trackers
                        last_ckpt = steps
                        best_model_score = model_score


                # update progress bar and counter
                prog_bar.update(update_step)
                steps += 1
            # self.scheduler.step()

        prog_bar.close()

        self.test()
        display.title('Finished Training')

