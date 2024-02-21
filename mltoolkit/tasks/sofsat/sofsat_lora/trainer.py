"""
this is an example trainer class that inherits TrainerBase
"""
# external imports
import torch
import numpy as np
import datasets
import peft
from torch import nn
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType

# for typing
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Tuple, List, Dict, TypeVar
T = TypeVar('T')

# local imports
from mltoolkit.templates import Trainer
from mltoolkit.utils import (
    files,
    strings,
    display,
    tensor_utils,
)
from . import data_module

class TrainerSofsatLora(Trainer):
    def __init__(self, config_path, debug=False, accelerator=None):
        super().__init__(
            config_path,
            debug,
            accelerator=accelerator
        )

    def setup(self):
        cfg = self.cfg

        train_loader, val_loader = data_module.get_dataloaders(cfg)

        model_name = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cfg.paths['cache'],
        )
        #self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.tokenizer.pad_token = self.tokenizer.eos_token

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=['q_proj', 'k_proj'],
            r=cfg.params['lora_r'],
            lora_alpha=cfg.params['lora_alpha'],
            lora_dropout=cfg.params['lora_dropout'],
        )
        model = peft.get_peft_model(base_model, peft_config)

        if self.accel.is_main_process:
            display.note(' ',end='')
            model.print_trainable_parameters()

        # optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.params['lr'],
            weight_decay=self.cfg.params['weight_decay']
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.params['sched_step_size'],
            gamma=cfg.params['sched_gamma'],
        )

        self.loss_fn = nn.CrossEntropyLoss()

        return {
            'sofsat-lora': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def _build_seqs(self, batch):

        op_tokens = {
            'op1_open': '<op1>',
            'op1_close': '</op1>',
            'op2_open': '<op2>',
            'op2_close': '</op2>',
            'union': '<union>',
            'intersection': '<intersection>',
            'right_diff': '<right_diff>',
            'left_diff': '<left_diff>',
        }

        seqs = ['<op1>' + s1  + '</op1>' + op_tokens[op] + '<op2>' + s2 + '</op2>' + sy
                for s1, s2, sy, op, 
                in zip(batch['S1'], batch['S2'], batch['Sy'], batch['operation'])]

        return seqs


    def step(self, batch: T, mode='train'):

        seqs = self._build_seqs(batch)

        tokens = self.tokenizer(seqs,
                                truncation=True,
                                max_length=self.cfg.params['seq_len']+1,
                                return_token_type_ids=False,
                                padding=True,
                                return_tensors='pt').to(self.accel.device)
        in_tokens = {key: val[:, :-1] for key, val in tokens.items()}
        labels = tokens['input_ids'][:, 1:]

        # model forward
        scores = self.train_vars['sofsat-lora'](**in_tokens)

        N, S, D = scores['logits'].shape

        loss = self.loss_fn(scores['logits'].reshape((-1, D)), labels.flatten())
        ppl = torch.exp(loss.detach())

        return loss, ppl

    def train_step(self, batch: T) -> Tuple[torch.Tensor, Dict]:
        """
        This function is used to compute the loss and training statistics

        Input
            model[nn.Module]: this is the model you previously assigned.
            batch[T]: a batch sampled from the previously assigned train dataloader

        Return
            loss[torch.Tensor]: the computed loss
            metrics[Dict]: the metrics you would like to track.
                refer to {project_root}/mltoolkit/trainers/base.py for the currently supported keyword trackers
        """

        loss, ppl = self.step(batch, mode='train')

        return loss, {
            'scalar' : {
                'loss' : loss,
                'perplexity': ppl,
            }
        }

    def eval_step(self, batch: T, loader_name):
        """
        same as train_step but batch comes from previously assigned val_loader (test_loader evaluation not implemented yet)
        NOTE: in this function, torch.Tensor values need to be converted to float/int 

        Input
            batch[T]: a batch sampled from the previously assigned train dataloader
            mode[str]: string that will be either 'val' or 'train' depending on the procedure. 

        Return
            metrics[Dict]: the metrics you would like to track. Each run of eval_step() will 
                accumulate these metrics into a Dataset object and will be used as input 
                to on_eval_end() for final aggregation
        """
        loss, ppl = self.step(batch, mode='val')

        return {
            'loss': float(loss),
            'perplexity': float(ppl),
        }

    def on_eval_end(self, metrics: List, mode: str):
        """
        use this to aggregate your metrics collected from the outputs of eval_step()

        Input:
            metrics[Dict[str, list]]: the set of metrics collected from eval_step()

        Return
            target_metric[T]: the metric that will be used to determine if the current model should be 
                checkpointed to {cfg.params['results']}/best_model.pt.
            log_values[Dict[Dict[str, T]]]: 
        """

        metrics_ds = Dataset.from_list(metrics['val_loader'])
        loss = np.mean(metrics_ds['loss'])
        ppl = np.mean(metrics_ds['perplexity'])

        return loss, {
            'scalar' : {
                'loss': loss,
                'perplexity': ppl,
            }
        }
