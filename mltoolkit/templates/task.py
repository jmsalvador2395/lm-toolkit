# external imports
import numpy as np
import itertools
import yaml
import json
import traceback
import os
from tqdm import tqdm
from copy import deepcopy
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from datasets import Dataset

# local imports
from mltoolkit.utils import (
    display,
    files,
)

class Task:

    def __init__(self, cfg, keywords, debug=False):
        self.cfg = cfg
        self.keywords = keywords
        self.debug = debug

        self.trainer_cls = None
        self.task_name = '<<Template>>'

    def train(self):
        if self.trainer_cls is None:
            display.error(f'Training procedure not defined for {self.task_name}')
            raise NotImplementedError()
        else:
            trainer = self.trainer_cls(self.cfg, self.debug)
            trainer.train()

    def param_search(self):
        if self.trainer_cls is None:
            display.error('self.trainer_cls needs to be assigned before search procedure can execute')
            raise ValueError()
        
        search_proc = self.cfg.search_params['search_type']
        if search_proc == 'grid':
            self._grid_search()
        elif search_proc == 'random':
            display.error(f'Random search procedure not implemented yet')
            raise NotImplementedError()

    def _grid_search(self):
        accel = Accelerator()
        cfg = self.cfg

        search_params = deepcopy(cfg.search_params)
        out_dir = cfg.paths['results'] + '/' + cfg.general['experiment_name']

        # pop the search settings
        step_limit = search_params.pop('step_limit')
        search_params.pop('search_type')

        # set save criterion
        save_criterion = cfg.params.get('save_criterion', None)

        # set initial score that's guaranteed to be overwritten
        best_score = float('-inf') if save_criterion == 'max' else float('inf')
        bad_score = best_score

        save_step = -1
        if save_criterion == 'max':
            criterion = lambda cur, prev: cur > prev
        elif save_criterion == 'min':
            criterion = lambda cur, prev: cur < prev
        else:
            display.error('cfg.params[save_criterion] not specified choose from: ["max", "min"]')
            raise ValueError()

        # build lists/arrays of search parameters
        for param in search_params.keys():
            vals = search_params[param]['values']
            n_samples = search_params[param].get('num_samples', None)
            dtype = search_params[param]['dtype']

            if n_samples is not None:
                param_arr = np.linspace(vals[0], vals[-1], n_samples)
            else:
                param_arr = np.array(vals)

            # set data types
            if dtype == 'int':
                param_arr = param_arr.astype(int)
            elif dtype == 'float':
                param_arr = param_arr.astype(float)

            search_params[param] = param_arr.tolist()

        # initialize tracking vars
        candidates = []
        scores = []

        params = list(search_params.keys())
        search_space = list(itertools.product(*search_params.values()))
        out_ds = []
        for param_set in search_space:
            out_ds.append({name: val for name, val in zip(params, param_set)})
        out_ds = Dataset.from_list(out_ds)

        if accel.is_main_process:
            display.title('Begin Gridsearch')
            prog_bar = tqdm(
                range(len(search_space)),
                desc='in gridsearch procedure',
            )

        # main loop for hyperparameter search
        for i, vals in enumerate(search_space):

            # build candidate config
            candidate_config = deepcopy(self.cfg)
            for j, param in enumerate(params):
                candidate_config.params[param] = vals[j]
            candidates.append(candidate_config)

            candidate_config.general['experiment_name'] = \
                candidate_config.general['experiment_name'] + f'/search-{i:03}'

            accel.wait_for_everyone()
            trainer = self.trainer_cls(candidate_config, accel)
            try:
                score = trainer.train(
                    step_limit=step_limit,
                    global_best_score=best_score,
                    exp_num=i,
                )
            except KeyboardInterrupt:
                display.done('Keyboard interrupt detected')
                os._exit(0)
            except Exception as e:
                display.error(
                    f'Exception occured during training. at parameter set {i}'
                )
                traceback.print_exception(e)
                os._exit(1)


            if accel.is_main_process:
                scores.append(score)
                temp_out = out_ds.select(range(i+1))
                temp_out = temp_out.add_column('score', scores)
                temp_out.to_json(out_dir + '/scores.json')

            # compare and save
            if accel.is_main_process and criterion(score, best_score):
                best_score = score
                save_step = i
                files.create_path(out_dir)
                with open(out_dir + '/best_config.yaml', 'w') as f:
                    candidate_config.general.pop('experiment_name')
                    f.write(yaml.dump(candidate_config._asdict()))
                with open(out_dir + '/info.txt', 'w') as f:
                    f.write(
                        f'best model score: {best_score}'
                        f'best model score found at step {save_step}'
                        f'params: {temp_out[-1]}'
                    )

            # update progbar
            if accel.is_main_process:
                prog_bar.set_postfix({
                    'best_score': best_score,
                    'save_step': save_step,
                })
                prog_bar.update()

        if accel.is_main_process:
            display.title('Finished Gridsearch')
            out_ds = out_ds.add_column('score', scores)
            if save_criterion == 'max':
                best_idx = np.argmax(out_ds['score'])
            else:
                best_idx = np.argmin(out_ds['score'])
            display.done(f'best score: {best_score}')
            print('best hyperparameters:')
            print(json.dumps(out_ds[int(best_idx)], indent=4))


    def evaluate(self):
        display.error(f'Evaluation procedure not defined for {self.task_name}')
        raise NotImplementedError()

    def other(self):
        display.error(f'Other procedure not defined for {self.task_name}')
        raise NotImplementedError()
