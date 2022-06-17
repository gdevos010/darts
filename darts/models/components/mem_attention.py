import math
from inspect import isfunction

import torch
from torch import nn as nn
from torch.nn import functional as F


def to(t):
    return {'dtype': t.dtype, 'device': t.device}


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]


def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim+1] = split_dims
    return t.reshape(shape)


def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()


class MemAttention(nn.Module):
    """
    Memory-based attention.
    source:[1]_

    References
    ----------
    .. [1] Memory-based Transformer with Shorter Window and Longer Horizon for Multivariate Time Series Forecasting(SWLHT)
        https://github.com/liuyang806/SWLHT/blob/main/models/attn.py
    """
    def __init__(self, mask_flag=True, factor=5, scale=None):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.attn_dropout = nn.Dropout(0.1)
        # self.dropout = nn.Dropout(0.1)
        # self.memory_attn_dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, attn_mask, pos_emb=None, kv_len=None, mem_len=None, lmem_len=None):
        B, H, L, E = q.shape
        scale = self.scale or 1. / math.sqrt(E)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * scale
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * scale
            pos_dots = shift(pos_dots)
            pos_dots = F.pad(pos_dots, (dots.shape[-1] - pos_dots.shape[-1], 0), value=0.)
            dots = dots + pos_dots

        # TODO
        # if attn_mask is not None:
        #     mask = attn_mask[:, None, :, None] * attn_mask[:, None, None, :]
        #     mask = F.pad(mask, (mem_len + lmem_len, 0), value=True)
        #     dots.masked_fill_(~mask, mask_value)

        total_mem_len = mem_len + lmem_len
        mask = torch.ones(L, L + total_mem_len, **to(q)).triu_(diagonal=1 + total_mem_len).bool()

        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return out



