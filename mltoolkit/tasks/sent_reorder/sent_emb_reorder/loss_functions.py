import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

def hinge_loss(scores, X, Y, mask, margin=1):
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

def hinge_pair_loss(scores, X, Y, mask, margin=1):

    reverse = torch.argsort(Y, dim=-1)
    unshuff_scores = scores[X, reverse]
    zero = torch.tensor(0.0, device=unshuff_scores.device)

    diffs = unshuff_scores[:, :-1] - unshuff_scores[:, 1:]
    loss = torch.max(zero, diffs+margin)
    loss = torch.mean(loss[mask[:, 1:]])

    return loss


def cross_entropy_loss(scores, X, Y, mask):

    sc = scores.masked_fill(~mask, float('-inf'))
    smax = F.softmax(sc, dim=-1)
    y = Y.to(torch.float).masked_fill(~mask, float('-inf'))
    y_smax = torch.softmax(y, dim=-1)
    loss = torch.mean(-torch.log(y_smax[mask]*smax[mask]))

    return loss

def huber_loss(scores, X, Y, mask):
    fn = nn.HuberLoss()
    loss = fn(scores[mask], Y.to(torch.float32)[mask])
    return loss

def pairwise_logistic_loss(scores, X, Y, mask):

    reverse = torch.argsort(Y, dim=-1)
    unshuf_scores = scores[X, reverse]
    loss = torch.log(
        1 + torch.exp(
            unshuf_scores[..., :-1]
            - unshuf_scores[..., 1:]
        )
    )
    loss = torch.mean(loss[mask[..., 1:]])
    return loss

def diff_kendall(
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

