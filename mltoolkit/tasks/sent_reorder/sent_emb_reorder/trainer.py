"""
this is an example trainer class that inherits TrainerBase
"""
# external imports
import torch
import numpy as np
import itertools
from torch import nn
from datasets import Dataset
from scipy.stats import kendalltau, spearmanr
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize

# for typing
from typing import Tuple, List, Dict, TypeVar
T = TypeVar('T')

# local imports
from mltoolkit.templates import Trainer
from mltoolkit.utils import tensor_utils
from mltoolkit.models.sent_encoders.hf_models import from_hf
from mltoolkit.utils import (
    files,
    strings,
    display,
)
from .model import SentEmbedReorder
from .data_module import get_dataloaders

class TrainerSentEmbedReordering(Trainer):
    def __init__(self, config_path, debug=False, accelerator=None):
        super().__init__(
            config_path,
            debug,
            accelerator=accelerator
        )

    def setup(self):
        cfg = self.cfg

        train_loader, val_loader = get_dataloaders(cfg)

        # define models
        model_name = "mixedbread-ai/mxbai-embed-large-v1"
        """
        encoder = from_hf(
            model_name, 
            emb_dim=1024, 
            max_seq_len=512,
            cache_dir=cfg.paths['cache'],
        )
        """
        encoder = AutoModel.from_pretrained(
            model_name,
            cache_dir=cfg.paths['cache'],
        )
        encoder.train()
        model = SentEmbedReorder(**cfg.params)

        tok = AutoTokenizer.from_pretrained(model_name)

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

        self.loss_fn = nn.HuberLoss()

        return {
            'reorder': model,
            'encoder': encoder,
            'tok': tok,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def step(self, batch: T, mode='train'):

        tok = self.train_vars['tok']

        # convert docs to batch of sentences
        sent_batch = [sent_tokenize(el) for el in batch['text']]
        N = len(sent_batch)

        # get number of sentences per doc
        sent_lengths = [len(doc) for doc in sent_batch]

        # convert (flatten) List[List[str]] to List[str]
        sent_batch_flt = list(itertools.chain.from_iterable(sent_batch))

        # tokenize
        tokens = tok(
                sent_batch_flt,
                truncation=True,
                max_length=self.cfg.params['sent_len'],
                return_token_type_ids=False,
                padding=True,
                return_tensors='pt').to(self.accel.device)

        # get sentence embeddings
        sent_embeds = \
            self.train_vars['encoder'](**tokens)['pooler_output']
        
        # group sentence embeddings by document
        embeds_per_doc = torch.split(sent_embeds, sent_lengths)
        embeds_per_doc = tensor_utils.pad_and_stack(embeds_per_doc)

        # create padding mask for embeds_per_doc
        embed_pad_mask = tensor_utils.get_pad_mask(
            embeds_per_doc.shape[:-1],
            np.array(sent_lengths),
            embeds_per_doc.device,
        )

        # get tensor dimensions
        N, L, D = embeds_per_doc.shape

        # create indexing arrays for shuffling
        X = np.arange(N)[None].T.repeat(L, axis=-1)
        Y = np.arange(L)[None].repeat(N, axis=0)

        # shuffle the indices while ignoring the padded dimensions
        for row, length in zip(range(N), sent_lengths):
            Y[row, :length] = np.random.permutation(length)

        # shuffle embeddings using the indices generated from above
        shuffled_embs = embeds_per_doc[X, Y]

        scores = self.train_vars['reorder'](shuffled_embs)
        labels = torch.tensor(
            Y, 
            dtype=torch.float32, 
            device=scores.device,
        ) 

        loss = self.loss_fn(
            scores,
            labels,
        )

        """
        preds = torch.argsort(scores)
        tau, p_tau = kendalltau(Y, preds.cpu().numpy())
        rho, p_rho = spearmanr(
            Y.flatten(), 
            preds.cpu().numpy().flatten()
        )
        """
        scores = scores.detach().cpu().numpy()
        kendall = [
            tuple(kendalltau(y, score))
            for y, score in zip(Y, scores)
            #tuple(kendalltau(y[msk], pred))
            #for y, msk, pred in zip(Y, mask, preds)
        ]
        tau, p_tau = zip(*kendall)
        tau = np.mean(tau)
        p_tau = np.mean(p_tau)

        spearman = [
            tuple(spearmanr(y, score))
            for y, score in zip(Y, scores)
            #tuple(spearmanr(y[msk], pred))
            #for y, msk, pred in zip(Y, mask, preds)
        ]
        rho, p_rho = zip(*spearman)
        rho = np.mean(rho)
        p_rho = np.mean(p_rho)

        return loss, tau, rho, p_tau, p_rho

    def train_step(self, batch: T) -> Tuple[torch.Tensor, Dict]:
        """
        This function is used to compute the loss and training 
        statistics

        Input
            model[nn.Module]: this is the model you previously assigned.
            batch[T]: a batch sampled from the previously assigned train 
                dataloader

        Return
            loss[torch.Tensor]: the computed loss
            metrics[Dict]: the metrics you would like to track.
                refer to {project_root}/mltoolkit/trainers/base.py for 
                the currently supported keyword trackers
        """

        loss, tau, rho, p_tau, p_rho = self.step(batch, mode='train')

        return loss, {
            'scalar' : {
                'loss' : loss,
                'tau': tau,
                'rho': rho,
                'p_tau': p_tau,
                'p_rho': p_rho,
            }
        }

    def eval_step(self, batch: T, loader_name):
        """
        same as train_step but batch comes from previously assigned 
        val_loader (test_loader evaluation not implemented yet)
        NOTE: in this function, torch.Tensor values need to be converted 
        to float/int 

        Input
            batch[T]: a batch sampled from the previously assigned train 
                dataloader
            mode[str]: string that will be either 'val' or 'train' 
            depending on the procedure. 

        Return
            metrics[Dict]: the metrics you would like to track. Each run 
                of eval_step() will accumulate these metrics into a 
                Dataset object and will be used as input to 
                on_eval_end() for final aggregation
        """
        #loss, accuracy = self.step(batch, mode='val')

        loss, tau, rho, p_tau, p_rho = self.step(batch, mode='val')

        return {
            'loss': float(loss),
            'tau': float(tau),
            'rho': float(rho),
            'p_tau': float(p_tau),
            'p_rho': float(p_rho),
        }

    def on_eval_end(self, metrics: List, mode: str):
        """
        use this to aggregate your metrics collected from the outputs of 
        eval_step()

        Input:
            metrics[Dict[str, list]]: the set of metrics collected from 
                eval_step()

        Return
            target_metric[T]: the metric that will be used to determine 
                if the current model should be checkpointed to 
                {cfg.params['results']}/best_model.pt.
            log_values[Dict[Dict[str, T]]]: 
        """

        metrics_ds = Dataset.from_list(metrics['val_loader'])
        loss = np.mean(metrics_ds['loss'])
        tau = np.mean(metrics_ds['tau'])
        rho = np.mean(metrics_ds['rho'])
        p_tau = np.mean(metrics_ds['p_tau'])
        p_rho = np.mean(metrics_ds['p_rho'])

        return rho, {
            'scalar' : {
                'loss' : loss,
                'tau': tau,
                'rho': rho,
                'p_tau': p_tau,
                'p_rho': p_rho,
            }
        }
