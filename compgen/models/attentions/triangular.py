import math
import torch
from torch import nn
from torch.nn import ModuleList
from opt_einsum import contract


class TriangularAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        bias: bool = False,
        mask_padding_value: float = -1e4,
        device: str = "cpu",
        use_dropout: bool = True
    ):
        super(TriangularAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaler = 1 / math.sqrt(self.head_dim)
        self.mask_padding_value = mask_padding_value
        self.device = device

        assert (
            self.hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads."

        self.w = ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=bias, dtype=dtype, device=device) for _ in range(5)
        ])
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
    

    def construct_mask(self, attention_mask):
        attention_mask = attention_mask
        new_mask = attention_mask.unsqueeze(2)+attention_mask.unsqueeze(3)
        new_mask = (new_mask+attention_mask.unsqueeze(1))
        new_mask = new_mask.unsqueeze(4)
        new_mask = new_mask.to(self.device)
        return new_mask


    def forward(self, hidden_state, batch_mask=None):        
        
        attention_mask = None
        if batch_mask is not None:
            attention_mask = batch_mask["attention_mask"]
        
        num_batches, num_nodes, _, _ = hidden_state.size()

        query = hidden_state
        key = hidden_state
        value = hidden_state

        left_k, right_k, left_v, right_v, query = [
            l(x) for l, x in zip(self.w, (key, key, value, value, key))
        ]

        left_k = left_k.view(
            num_batches, num_nodes, num_nodes, self.num_heads, self.head_dim
        )
        right_k = right_k.view_as(left_k)
        left_v = left_v.view_as(left_k)
        right_v = right_v.view_as(left_k)
        query = query.view_as(left_k)

        scores = contract("bxahd,bayhd->bxayh", left_k, right_k) * self.scaler

        if attention_mask is not None:
            expanded_mask = self.construct_mask(attention_mask)
            scores = scores.masked_fill(expanded_mask, self.mask_padding_value)

        val = contract("bxahd,bayhd->bxayhd", left_v, right_v)

        # Improve numerical stability
        scores = scores - scores.max(dim=2, keepdim=True).values

        att_weights = scores.softmax(dim=2)
        if self.use_dropout:
            att_weights = self.dropout(att_weights)

        att = contract("bxayh,bxayhd->bxyhd", att_weights, val)
        att = att.view(num_batches, num_nodes, num_nodes, self.hidden_dim)

        return att, att_weights
