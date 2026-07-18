import torch
import torch.nn as nn

from models.attentions.smarter import SmarterAttention
from models.attentions.standard import StandardAttention
from models.attentions.third_order import ThirdOrderAttention
from models.attentions.triangular import TriangularAttention

class BackboneAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention_type: str = "standard",
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        use_norm: bool = False,
        dtype: torch.dtype = torch.float64,
        bias: bool = False,
        mask_padding_value: float = -1e4,
        device: str = "cpu",
        is_edge: bool = False,
        use_dropout: bool = True,
    ):

        super(BackboneAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.bias = bias
        self.attention_type = attention_type
        self.mask_padding_value = mask_padding_value
        self.device = device
        self.is_edge = is_edge
        self.use_dropout = use_dropout

        # Dictionary of available attention classes
        attention_classes = {
            "standard": StandardAttention,
            "smarter": SmarterAttention,
            "third_order": ThirdOrderAttention,
            "triangular": TriangularAttention,
        }

        # Validate and instantiate the selected attention type
        if attention_type not in attention_classes:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        attention_class = attention_classes[attention_type]

        self.attention = attention_class(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dtype=dtype,
            bias=bias,
            mask_padding_value=mask_padding_value,
            device=device,
            use_dropout=use_dropout,
        )

        self.init_weight()

    def forward(self, hidden_state, batch_mask=None):
        # Compute attention output
        if not self.is_edge:
            attention_output = self.attention(
                hidden_state=hidden_state, batch_mask=batch_mask
            )
        else:
            attention_output = self.attention(
                hidden_state=hidden_state, batch_mask=batch_mask
            )

        return attention_output

    def init_weight(self):
        for name, param in self.attention.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
            else:
                raise ValueError(f"Unsupported parameter: {name}")
