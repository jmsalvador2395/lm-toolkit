# external imports
import torch
import numpy as np
import datasets

# local imports
from mltoolkit.utils import (
    files,
    strings,
    display,
    data
)
class Trainer():
    
    def __init__(self, cfg):
        self.model = None
        self.cfg = cfg

        # seed the random number generators
        torch.manual_seed(cfg.general['seed'])
        np.random.seed(cfg.general['seed'])

        # load datasets
        self.ds_train, self.ds_val = data.load_ds(cfg.data)

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
    def _log_hparams(name, writer, params):
        category = 'hparams'

        # create markdown table
        table = \
            '# Training Parameters\n' + \
            '| parameter | value |\n' + \
            '| --------- | ----- |\n'
        for key, val in sorted(params.items()):
            table += f'| {key} | {val} |\n'

        # write table to tensorboard
        writer.add_text(category, table)

    def train_step(self, batch):
        pass

    def val_step(self, batch):
        pass

    def validate(self):
        pass

    def train(self):
        cfg = self.cfg
        # seed the random number generators

        # set experiment name
        experiment_name = f'{strings.now()}-{cfg.model["name"]}'

        # ccreate checkpoint directory
        ckpt_dir = cfg.training['ckpt_dir'] + f'/{experiment_name}/'
        files.create_path(ckpt_dir)


        # load the dataset
        ds = data.load_corpus(corpus_dir, cache_dir=cache_dir, num_proc=num_proc)

        # pre-process dataset using mapping function
        print(strings.green('\nPreparing dataset for training...'))
        breakpoint()

        ds, mapping_params, model_params = mapper(ds)
        params.update(mapping_params)

        # initialize task
        task = task_manager.get(task_name, params, session_tools)

        misc.print_title('Begin Training')
        prog_bar = tqdm(
            range(len(ds['train'])*num_epochs),
            desc=experiment_name
        )
        # used for checkpoint names
        checkpoint_log_digits = len(str(len(ds['train'])*num_epochs))
        steps = 0

        # initialize tensorboard logger and log hyperparameters
        log_dir = f'{misc.get_project_root()}/tensorboard/{experiment_name}'
        writer = SummaryWriter(
            log_dir=log_dir
        )
        log_hparams(experiment_name, writer, params)

        # break dataset into shards
        shard_ids = np.arange(num_shards)
        if shuffle:
            np.random.shuffle(shard_ids)

        for epoch, shard in product(range(num_epochs), shard_ids):

            # split get shard and shuffle if specified
            ds_shard = ds['train'].shard(num_shards=num_shards, index=shard)
            if shuffle:
                generator = np.random.default_rng(seed=params['seed'])
                ds_shard = ds_shard.shuffle(generator=generator)

            #for batch in DataLoader(ds_shard, batch_size=batch_size, shuffle=shuffle):
            for i in range(0, len(ds_shard), batch_size):
                batch = ds_shard[i:i+batch_size]

                # calculate loss
                loss, loss_stats = task.loss(batch)

                # optimzer step
                task.optimize(loss)
                
                # update tracked parameters
                prog_bar.set_postfix({
                    'loss' : f'{loss.detach().cpu().numpy():.02f}',
                    'epoch' : epoch,
                    'step' : steps,
                })

                # log stats to tensorboard
                if steps % eval_freq == 0:
                    metrics = task.evaluate()
                    metrics['scalar'].update({
                        'loss/train' : loss.detach().cpu().numpy(),
                        'epoch' : epoch
                    })
                    metrics['scalar'].update(loss_stats)
                    log(writer, metrics, steps)

                if steps % ckpt_freq == 0:
                    torch.save(task.model, f'{ckpt_dir}step{steps:0{checkpoint_log_digits}}.pth')

                # update progress bar and counter
                prog_bar.update(batch_size)
                steps += 1

        torch.save(task.model, f'{ckpt_dir}step{steps:0{checkpoint_log_digits}}.pth')
        misc.print_title('Finished Training')


