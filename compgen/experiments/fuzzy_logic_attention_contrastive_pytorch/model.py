"""PyTorch port of the reference standard-softmax-attention Transformer."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _relative_position_bucket(relative_position, num_buckets, max_distance):
    # T5-style, unidirectional bucketing used by the reference configuration.
    n = torch.maximum(-relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    large = max_exact + (
        torch.log(n.float().clamp_min(1) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = torch.minimum(large, torch.full_like(large, num_buckets - 1))
    return torch.where(is_small, n, large)


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, num_buckets=16, max_distance=16):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, length, device):
        context = torch.arange(length, device=device)[:, None]
        memory = torch.arange(length, device=device)[None, :]
        buckets = _relative_position_bucket(memory - context, self.num_buckets, self.max_distance)
        return self.embedding(buckets).permute(2, 0, 1).unsqueeze(0)


class StandardAttention(nn.Module):
    def __init__(self, emb_dim, qk_dim, v_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.qk_head_dim = qk_dim // num_heads
        self.v_head_dim = v_dim // num_heads
        self.query = nn.Linear(emb_dim, qk_dim)
        self.key = nn.Linear(emb_dim, qk_dim)
        self.value = nn.Linear(emb_dim, v_dim)
        self.out = nn.Linear(v_dim, emb_dim)
        self.dropout = dropout

    def forward(self, x, bias):
        batch, length, _ = x.shape
        q = self.query(x).view(batch, length, self.num_heads, self.qk_head_dim)
        k = self.key(x).view(batch, length, self.num_heads, self.qk_head_dim)
        v = self.value(x).view(batch, length, self.num_heads, self.v_head_dim)
        scores = torch.einsum("bqhd,bkhd->bhqk", q / math.sqrt(self.qk_head_dim), k)
        weights = torch.softmax(scores + bias, dim=-1)
        if self.training and self.dropout:
            # Reference broadcasts one attention-dropout mask across batch and heads.
            mask = torch.empty(1, 1, length, length, device=x.device).bernoulli_(1 - self.dropout)
            weights = weights * mask / (1 - self.dropout)
        values = torch.einsum("bhqk,bkhd->bqhd", weights, v).flatten(2)
        return self.out(values), weights


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.emb_dim, eps=1e-6)
        self.attention = StandardAttention(
            config.emb_dim, config.qk_dim, config.v_dim,
            config.num_heads, config.attention_dropout_rate,
        )
        self.relative_bias = RelativePositionBias(
            config.num_heads, config.seq_len, config.seq_len
        )
        self.norm2 = nn.LayerNorm(config.emb_dim, eps=1e-6)
        self.mlp1 = nn.Linear(config.emb_dim, config.mlp_dim)
        self.mlp2 = nn.Linear(config.mlp_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        mixed, attention = self.attention(
            self.norm1(x), self.relative_bias(x.shape[1], x.device)
        )
        x = x + self.dropout(mixed)
        y = self.mlp2(self.dropout(F.gelu(self.mlp1(self.norm2(x)), approximate="tanh")))
        return x + self.dropout(y), attention


class StandardAttentionTransformer(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.embedding = nn.Linear(input_dim, config.emb_dim)
        self.input_dropout = nn.Dropout(config.dropout_rate)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.emb_dim, eps=1e-6)
        self.output = nn.Linear(config.emb_dim, 1)
        self.apply(self._init_reference_style)

    @staticmethod
    def _init_reference_style(module):
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            std = math.sqrt(1.0 / fan_in) / 0.87962566103423978
            nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=math.sqrt(1.0 / module.weight.shape[0]))

    def forward(self, x, return_attention=False):
        x = self.input_dropout(self.embedding(x))
        attentions = []
        for block in self.blocks:
            x, attention = block(x)
            attentions.append(attention)
        prediction = self.output(self.final_norm(x))
        return (prediction, attentions) if return_attention else prediction
