# external imports
import torch
import transformers
import math
from torch import nn

# local imports
from mltoolkit.utils import (
    display,
    tensor_utils
)

class MeanPooling(nn.Module):

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, input_ids, attention_mask):
        #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(input_ids.size()).float()
        return (
            torch.sum(input_ids * input_mask_expanded, 1) 
            / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

class AttentionPooling(nn.Module):
    def __init__(self, 
                 d_input,
                 d_attention,
                 n_pool_heads,
                 dropout=0.1,
                 **kwargs):

        super(AttentionPooling, self).__init__()
        assert d_attention % n_pool_heads == 0, 'number of heads does not evenly divide attention vectors'
        
        self.d = d_attention
        self.n_heads = n_pool_heads
        self.qk_proj = nn.Linear(d_input, d_attention*2)
        self.drop = nn.Dropout(dropout)

    def forward(self, v, attention_mask):
        """
        attention mask gets used as pad mask here
        """

        q, k = self.qk_proj(v).chunk(2, dim=-1)

        N, S, D = q.shape
        _, T, _ = v.shape
        H = self.n_heads

        q = q.view((N, S, H, D//H)).permute(0, 2, 1, 3)
        k = k.view((N, T, H, D//H)).permute(0, 2, 3, 1)
        v = v.view((N, T, H, D//H)).permute(0, 2, 1, 3)

        result = (q@k)/math.sqrt(D)

        mask = (attention_mask[:, None, :].repeat(1, H, 1) == 0)
        #result = result.masked_fill(mask, float('-inf'))
        result = result.masked_fill(mask[..., None], float('-inf'))
        #result = result.masked_fill(mask[..., None, :], float('-inf'))
        result = self.drop(torch.softmax(result, dim=-1))

        result = (result@v).movedim(2, 1)
        result = result.reshape((N, T, -1))
        #result[torch.isnan(result)] = 0
        return result[:, 0, :]
