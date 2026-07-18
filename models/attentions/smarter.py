import math
import torch
import torch.nn as nn
from opt_einsum import contract


class SmarterAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=1,
        dropout_rate=0.0,
        dtype=torch.float64,
        bias=False,
        mask_padding_value: float = -1e4,
        device: str = "cpu",
        use_dropout: bool = True,
    ):
        super(SmarterAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaler = 1 / math.sqrt(self.head_dim)
        self.mask_padding_value = mask_padding_value
        self.device = device
        self.dtype = dtype

        assert (
            self.hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads."

        self.w = nn.Linear(
            self.hidden_dim, self.hidden_dim * 5, bias=bias, dtype=dtype, device=device
        )
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    
    def construct_mask(self, attention_mask):
        attention_mask = attention_mask.flatten(1)
        expanded_mask = attention_mask.unsqueeze(1) + attention_mask.unsqueeze(2)
        expanded_mask = expanded_mask.unsqueeze(1)
        expanded_mask = expanded_mask.to(self.device)
        return expanded_mask
    
    def forward(self, hidden_state, batch_mask=None):

        attention_mask = None
        if batch_mask is not None:
            attention_mask = batch_mask["attention_mask"]

        B, N, C = hidden_state.shape
        H = self.num_heads
        D = self.head_dim

        w = self.w(hidden_state).reshape(B, N, 5, H, D).permute(2, 0, 3, 1, 4)
        a, b, c, v1, v2 = w[0], w[1], w[2], w[3], w[4]

        X = contract("bhid,bhjd->bhij", a, b) * self.scaler
        Y = contract("bhjd,bhkd->bhjk", b, c) * self.scaler
        Z = contract("bhkd,bhid->bhki", c, a) * self.scaler


        if attention_mask is not None:
            expanded_mask = self.construct_mask(attention_mask)

            X = X.masked_fill(expanded_mask, self.mask_padding_value)
            Y = Y.masked_fill(expanded_mask, self.mask_padding_value)
            Z = Z.masked_fill(expanded_mask, self.mask_padding_value)

        X = X - torch.max(X, dim=-1, keepdim=True).values
        Y = (
            Y
            - torch.max(
                torch.max(Y, dim=-1, keepdim=True).values, dim=-2, keepdim=True
            ).values
        )
        Z = Z - torch.max(Z, dim=-2, keepdim=True).values

        X = X.exp()
        Y = Y.exp()
        Z = Z.exp()

        if self.use_dropout:
            X = self.dropout(X)
            Y = self.dropout(Y)
            Z = self.dropout(Z)
        
        V = contract("bhjd,bhkd->bhjkd", v1, v2)

        up = contract("bhikd,bhki->bhid", contract("bhij,bhjk,bhjkd->bhikd", X, Y, V), Z)
        down = contract("bhik,bhki->bhi", contract("bhij,bhjk->bhik", X, Y), Z)
        down = down + 1e-9

        att = up / down.unsqueeze(-1)
        
        att = att.transpose(1, 2)
        att = att.reshape(B, N, C)

        return att, None
