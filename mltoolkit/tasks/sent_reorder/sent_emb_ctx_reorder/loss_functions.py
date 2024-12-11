import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple

def hinge_pair_plus_huber(
    scores, 
    X, 
    Y, 
    mask, 
    margin=1, 
    **kwargs
) -> Tuple[Tensor, Tensor]:
    l1 = hinge_pair_loss(scores, X, Y, mask, margin=0)
    l2 = huber_loss(scores, X, Y, mask, **kwargs)

    return l1 + l2

def hinge_loss(
    scores, 
    X, 
    Y, 
    mask, 
    margin=1, 
    **kwargs
) -> Tuple[Tensor, Tensor]:
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

        losses = torch.max(zero, margin+lower-upper)
        sum += torch.sum(losses[msk])

    loss = sum/total
    return loss

def hinge_pair_loss(
    scores, 
    X, 
    Y, 
    mask, 
    margin=1, 
    mask_zeros=False,
    **kwargs
) -> Tensor:

    reverse = torch.argsort(Y, dim=-1)
    unshuff_scores = scores[X, reverse]
    zero = torch.tensor(0.0, device=unshuff_scores.device)

    diffs = unshuff_scores[:, :-1] - unshuff_scores[:, 1:]
    loss = torch.max(zero, diffs+margin)

    #loss = torch.mean(loss[mask[:, 1:]])
    loss = loss[mask[:, 1:]]
    if mask_zeros and kwargs['mode'] == 'train':
        loss = loss[loss > 0]
    loss = torch.mean(loss)

    # add center term to keep the mean of scores close to 0
    #center = torch.mean(scores*mask, dim=-1)
    #center_loss = nn.HuberLoss()(center, torch.ones_like(center))
    #return loss+center_loss

    return loss

def masked_hinge_pair_loss(
    scores, 
    X, 
    Y, 
    mask, 
    margin=1, 
    **kwargs
) -> Tensor:

    reverse = torch.argsort(Y, dim=-1)
    #unshuff_scores = scores[X, reverse]
    unshuff_scores = scores[X, Y]
    zero = torch.tensor(0.0, device=unshuff_scores.device)
    preds = torch.argsort(scores, dim=-1)

    # use wrong_mask to train on wrong placements only
    wrong_mask = (Y != preds)
    #wrong_mask = wrong_mask[X, reverse]

    left = torch.cumsum(wrong_mask, dim=-1)
    right = torch.fliplr(torch.cumsum(torch.fliplr(wrong_mask), dim=-1))
    train_mask = (left*right).to(torch.bool)
    train_mask = train_mask[:, :-1] & train_mask[:, 1:]
    #train_mask = wrong_mask[:, :-1] | wrong_mask[:, 1:]

    diffs = unshuff_scores[:, :-1] - unshuff_scores[:, 1:]
    loss = torch.max(zero, diffs+margin)*train_mask

    loss = torch.mean(loss[mask[:, 1:]])

 
def masked_hinge_loss(
    scores, 
    X, 
    Y, 
    mask, 
    margin=1, 
    **kwargs
) -> Tuple[Tensor, Tensor]:
    rows, cols = X.shape
    total = torch.sum(mask)
    unshuffled_scores = scores[X, Y]
    zero = torch.tensor(0.0, device=unshuffled_scores.device)

    preds = torch.argsort(scores, dim=-1)
    wrong_mask = (Y != preds)
    # left = torch.cumsum(wrong_mask, dim=-1)
    # right = torch.fliplr(torch.cumsum(torch.fliplr(wrong_mask), dim=-1))
    # train_mask = (left*right).to(torch.bool)
    train_mask = wrong_mask

    sum = torch.tensor(0.0, device=scores.device)
    for i in range(cols-1):
        lower = unshuffled_scores[:, i, None]
        upper = unshuffled_scores[:, i+1:]
        msk = mask[:, i+1:]

        losses = torch.max(zero, margin+lower-upper)
        losses *= (train_mask[:, i, None] & train_mask[:, i+1:])
        sum += torch.sum(losses[msk])

    loss = sum/total

    # 
    loss += torch.mean(scores)**2
    return loss

def cross_entropy_loss(scores, X, Y, mask, **kwargs):

    sc = scores.masked_fill(~mask, float('-inf'))
    smax = F.softmax(sc, dim=-1)
    y = Y.to(torch.float).masked_fill(~mask, float('-inf'))
    y_smax = torch.softmax(y, dim=-1)
    loss = torch.mean(-torch.log(y_smax[mask]*smax[mask]))

    return loss

def huber_loss(scores, X, Y, mask, scale=1, zero_mean=True, **kwargs):
    fn = nn.HuberLoss()
    if zero_mean:
        mean = (
            torch.sum(Y*mask, dim=-1, keepdim=True)
            / torch.sum(mask, dim=-1, keepdim=True)
        )
        loss = fn(scores[mask], scale*(Y.to(torch.float32)-mean)[mask])
    else:
        loss = fn(scores[mask], scale*Y.to(torch.float32)[mask])

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
    **kwargs,
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
    
    return -torch.mean((1/N_0)*sums)

def hinge_pair_plus_diff_kendall(
    scores: Tensor, 
    X: Tensor, 
    Y: Tensor, 
    mask: Tensor,
    weight=.5,
    **kwargs,
) -> Tensor:
    term1 = hinge_pair_loss(scores, X, Y, mask, **kwargs)
    term2 = diff_kendall(scores, X, Y, mask, **kwargs)
    return weight*term1 + (1-weight)*term2

def exclusive(
    scores: Tensor, 
    X: Tensor, 
    Y: Tensor, 
    mask: Tensor,
    alpha: Tensor=.1,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    N = len(scores)
    loss = torch.tensor(0.0, device=scores.device)
    for score, y, msk in zip(scores, Y, mask):
        n = torch.sum(msk)
        sc = score[:n, :n]
        op = torch.log_softmax(sc, dim=-1)
        po = torch.log_softmax(sc.T, dim=-1)

        loss += -torch.mean((op + po)[range(n), y])
    loss /= N
    return loss, torch.argmax(scores, dim=-1)
