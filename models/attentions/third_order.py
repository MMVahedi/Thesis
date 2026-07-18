import math
import torch
import torch.nn as nn
from opt_einsum import contract


class ThirdOrderAttention(nn.Module):
    def __init__(
        self, hidden_dim, num_heads=1, dropout_rate=0.0, dtype=torch.float64, 
        bias=False, mask_padding_value: float = -1e4, device: str = "cpu", use_dropout: bool = True
    ):
        super(ThirdOrderAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaler = 1 / math.sqrt(self.head_dim)
        self.mask_padding_value = mask_padding_value
        self.device = device

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads."

        self.w = nn.Linear(self.hidden_dim, self.hidden_dim * 5, bias=bias, dtype=dtype, device=device)
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
    
    def construct_mask(self, attention_mask):
        attention_mask = attention_mask.flatten(1).to(self.device)
        new_mask = attention_mask.unsqueeze(2)+attention_mask.unsqueeze(1)
        new_mask = (new_mask.unsqueeze(3)+attention_mask.unsqueeze(1).unsqueeze(2))
        new_mask = new_mask.unsqueeze(1)
        new_mask = new_mask.flatten(-2)
        return new_mask
    
    def forward(self, hidden_state, batch_mask=None):
        
        attention_mask = None
        if batch_mask is not None:
            attention_mask = batch_mask["attention_mask"]

        B, N, C = hidden_state.shape
        NN = N**2
        H = self.num_heads
        D = self.head_dim

        wx = self.w(hidden_state).reshape(B, N, 5, H, D).permute(2, 0, 3, 1, 4)
        qi, kj, kk, vj, vk = (
            wx[0],
            wx[1],
            wx[2],
            wx[3],
            wx[4],
        )

        Kjk = contract("bhjd,bhkd->bhjkd", kj, kk)

        scores = contract("bhid,bhjkd->bhijk", qi, Kjk)
        scores = scores.reshape(B, H, N, NN)
        scores = scores * self.scaler

        if attention_mask is not None:
            expanded_mask = self.construct_mask(attention_mask)            
            scores = scores.masked_fill(
                expanded_mask, self.mask_padding_value   
            )
        
        scores = scores - scores.max(dim=-1, keepdim=True).values
        att_weights = torch.softmax(scores, -1)
        if self.use_dropout:
            att_weights = self.dropout(att_weights)

        att_weights = att_weights.reshape(B, H, N, N, N)

        Vjk = contract("bhjd,bhkd->bhjkd", vj, vk)

        att = contract("bhijk,bhjkd->bhid", att_weights, Vjk)
        att = att.transpose(1, 2)
        att = att.reshape(B, N, C)

        return att, None
