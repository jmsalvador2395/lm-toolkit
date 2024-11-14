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
from torch import Tensor

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
        model_name = cfg.params['encoder']
        self.model_name = model_name
        encoder = AutoModel.from_pretrained(
            model_name,
            cache_dir=cfg.paths['cache'],
        )

        # freeze encoder if specified
        if cfg.params['freeze_encoder']:
            encoder.eval()

        model = SentEmbedReorder(**cfg.params)

        tok = AutoTokenizer.from_pretrained(model_name)

        # optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.params['lr'],
            weight_decay=self.cfg.params['weight_decay']
        )

        enc_optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=self.cfg.params['encoder_lr'],
            weight_decay=self.cfg.params['encoder_weight_decay'],
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
            'enc_optimizer': enc_optimizer,
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    def hinge_loss(self, scores, X, Y, mask, margin=1):
        rows, cols = X.shape
        total = torch.sum(mask)
        reverse = torch.argsort(Y, dim=-1)
        unshuffled_scores = scores[X, reverse]
        zero = torch.tensor(0.0, device=unshuffled_scores.device)

        sum = torch.tensor(0.0, device=scores.device)
        for i in range(cols-1):
            lower = unshuffled_scores[:, i, None]
            upper = unshuffled_scores[:, i+1:]
            msk = mask[:, i+1:]

            losses = torch.max(zero, lower-upper+margin)
            sum += torch.sum(losses[msk])

        loss = sum/total
        return loss

    def cross_entropy_loss(self, scores, X, Y, mask):
        # TODO finish calculating this with a mask
        # TODO make this a function.
        exps = torch.exp(scores)
        exps = torch.masked_fill(~mask, float('-inf'))
        exp_sums = torch.sum(exps, axis=1)[:, None]
        softmax = exps/exp_sums
        loss = torch.mean(-torch.log(softmax[mask]))

        return loss

    def pairwise_logistic_loss(self, scores, X, Y, mask):

        unshuf_scores = scores[X, Y]
        loss = torch.log(
            1 + torch.exp(
                unshuf_scores[..., :-1]
                - unshuf_scores[..., 1:]
            )
        )
        loss = torch.mean(loss[mask[..., 1:]])
        return loss
    
    def diff_kendall(self,
        scores: Tensor, 
        X: Tensor, 
        Y: Tensor, 
        mask: Tensor,
        alpha: Tensor=.1,
    ) -> Tensor:
        dev = scores.device
        rows, cols = scores.shape
        N_0 = torch.sum(mask, dim=-1)
        sums = torch.zeros(rows, device=dev)
        for i in range(1, cols):
            for j in range(i):

                term1 = torch.exp(alpha*(scores[:, i] - scores[:, j]))
                term2 = torch.exp(-alpha*(scores[:, i] - scores[:, j]))
                term3 = torch.exp(alpha*(Y[:, i] - Y[:, j]))
                term4 = torch.exp(-alpha*(Y[:, i] - Y[:, j]))

                frac1 = (term1 - term2)/(term1 + term2)
                frac2 = (term3 - term4)/(term3 + term4)

                sums += (frac1*frac2)*(mask[:, i]*mask[:, j])
        
        return torch.mean((1/N_0)*sums)

    def step(self, batch: T, mode='train'):

        tok = self.train_vars['tok']
        seq_len = self.cfg.params['seq_len']

        # convert docs to batch of sentences
        # NOTE: this also truncates each document down to 
        #       `seq_len` sentences
        sent_batch = [
            sents[:seq_len] for sents in batch['sentences']
        ]
        #sent_batch = [text.split('<eos>') for text in batch['text']]
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
                return_tensors='pt'
        ).to(self.accel.device)

        # get sentence embeddings
        if self.cfg.params['freeze_encoder']:
            with torch.no_grad():
                sent_embeds = \
                    self.train_vars['encoder'](**tokens)
        else:
            sent_embeds = \
                self.train_vars['encoder'](**tokens)

        if 'pooler_output' in sent_embeds.keys():
            sent_embeds = sent_embeds['pooler_output']
        elif self.model_name == 'facebook/bart-large':
            sent_embeds = \
                sent_embeds['encoder_last_hidden_state'][:, 0, :]
        else:
            sent_embeds = sent_embeds['last_hidden_state'][:, 0, :]
        
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
        #X = np.arange(N)[None].T.repeat(L, dim=-1)
        #Y = np.arange(L)[None].repeat(N, dim=0)
        X = torch.arange(N)[None].T.repeat((1, L))
        Y = torch.arange(L)[None].repeat((N, 1))

        # shuffle the indices while ignoring the padded dimensions
        for row, length in zip(range(N), sent_lengths):
            #Y[row, :length] = np.random.permutation(length)
            Y[row, :length] = torch.randperm(length, device=Y.device)

        # shuffle embeddings using the indices generated from above
        shuffled_embs = embeds_per_doc[X, Y]

        # create mask to train on specific outputs
        label_mask = ~embed_pad_mask

        # compute scores
        scores = self.train_vars['reorder'](
            shuffled_embs, 
            embed_pad_mask
        )
        scores_masked = scores[label_mask]

        # set labels
        labels = Y.clone().detach().to(torch.float32).to(scores.device)
        labels = labels[label_mask]

        # compute loss
        """
        loss = self.loss_fn(
            scores_masked,
            labels,
        )
        """
        #loss = self.pairwise_logistic_loss(scores, X, Y, label_mask)
        """
        loss = -self.diff_kendall(
            scores, X.to(scores.device), 
            Y.to(scores.device), label_mask,
        )
        """
        loss = self.hinge_loss(
            scores, X.to(scores.device),
            Y.to(scores.device), label_mask,
        )

        # convert tensors to numpy
        scores = scores.detach().cpu().numpy()
        Y = Y.cpu().numpy()
        label_mask = label_mask.cpu().numpy()

        kendall = [
            tuple(kendalltau(y[mask], score[mask]))
            for y, score, mask in zip(Y, scores, label_mask)
        ]
        tau, p_tau = zip(*kendall)
        tau = np.mean(tau)
        p_tau = np.mean(p_tau)
        """
        import math
        if math.isnan(tau):
            breakpoint()
        breakpoint()
        """

        spearman = [
            tuple(spearmanr(y[mask], score[mask]))
            for y, score, mask in zip(Y, scores, label_mask)
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
