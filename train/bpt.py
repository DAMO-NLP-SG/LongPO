import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc_in = nn.Linear(dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.resid_dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(dim)

    def forward(self, hidden_states):
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.resid_dropout(hidden_states)
        return hidden_states

    def checkpoint_fn(self, carry, x):
        return None, checkpoint(self.forward, x)

def blockwise_compute_ffn(cell, inputs, chunk_size):
    inputs = rearrange(inputs, 'b (n c) d -> b n c d', c=chunk_size)
    inputs = rearrange(inputs, 'b n c d -> n b c d')
    _, res = scan(cell.checkpoint_fn, None, inputs, inputs.shape[0])
    res = rearrange(res, 'n b c d -> b (n c) d')
    return res