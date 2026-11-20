import torch
import torch.nn as nn


class ChannelLayerNorm3d(nn.Module):
    """Apply LayerNorm over channels independently for each space-time token."""

    def __init__(self, channels):
        super(ChannelLayerNorm3d, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3)


class ResidualCapacityBlock(nn.Module):
    """Parameter-capacity control with no content-dependent attention."""

    def __init__(self, channels, expansion=2, dropout_rate=0.0, residual_scale_init=0.0):
        super(ResidualCapacityBlock, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")

        expanded_channels = channels * expansion
        self.norm = ChannelLayerNorm3d(channels)
        self.expand = nn.Conv3d(channels, expanded_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.project = nn.Conv3d(expanded_channels, channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x):
        residual = self.project(self.dropout(self.activation(self.expand(self.norm(x)))))
        return x + self.residual_scale * residual
