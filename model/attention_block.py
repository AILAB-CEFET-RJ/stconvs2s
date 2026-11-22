import torch
import torch.nn as nn

from .capacity_block import ChannelLayerNorm3d


class ResidualTemporalSelfAttentionBlock(nn.Module):
    """Temporal self-attention independently at each spatial location."""

    def __init__(
        self, channels, sequence_length, num_heads=4, dropout_rate=0.0,
        residual_scale_init=0.0,
    ):
        super(ResidualTemporalSelfAttentionBlock, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.norm = ChannelLayerNorm3d(channels)
        self.horizon_queries = nn.Parameter(
            torch.empty(1, sequence_length, channels),
        )
        nn.init.normal_(self.horizon_queries, mean=0.0, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError("input must have shape [batch, channels, time, height, width]")

        batch, channels, steps, height, width = x.shape
        if steps != self.sequence_length:
            raise ValueError("input time dimension must match sequence_length")
        tokens = self.norm(x).permute(0, 3, 4, 2, 1).reshape(
            batch * height * width, steps, channels,
        )
        queries = tokens + self.horizon_queries
        attended, _ = self.attention(queries, tokens, tokens, need_weights=False)
        attended = attended.reshape(batch, height, width, steps, channels)
        attended = attended.permute(0, 4, 3, 1, 2).contiguous()
        return x + self.residual_scale * attended
