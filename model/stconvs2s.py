import torch
import torch.nn as nn

from .temporal_block import TemporalReversedBlock, TemporalCausalBlock
from .spatial_block import SpatialBlock
from .generator_block import TemporalGeneratorBlock
from .capacity_block import ResidualCapacityBlock
from .attention_block import ResidualTemporalSelfAttentionBlock


class STConvS2S_R(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5, output_channels=None):
        super(STConvS2S_R, self).__init__()
        
        self.stconvs2s_r = Model(TemporalReversedBlock, input_size, num_layers, hidden_dim, 
                                 kernel_size, device, dropout_rate, step, output_channels)
                                            
    def forward(self, x):
        return self.stconvs2s_r(x)

        
     
class STConvS2S_C(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5, output_channels=None):
        super(STConvS2S_C, self).__init__()
        
        initial_in_channels = input_size[1]
        input_length = input_size[2]
        
        self.stconvs2s_c = Model(TemporalCausalBlock, input_size, num_layers, hidden_dim, 
                                 kernel_size, device, dropout_rate, step, output_channels)
                                            
    def forward(self, x):
        return self.stconvs2s_c(x)        


class STConvS2S_C_CapacityControl(nn.Module):
    """STConvS2S-C with a parameter-matched residual capacity control."""

    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate,
                 step=5, output_channels=None, capacity_expansion=2):
        super(STConvS2S_C_CapacityControl, self).__init__()

        self.stconvs2s_c = Model(
            TemporalCausalBlock, input_size, num_layers, hidden_dim, kernel_size, device,
            dropout_rate, step, output_channels,
            feature_adapter_factory=lambda channels: ResidualCapacityBlock(
                channels, expansion=capacity_expansion, dropout_rate=dropout_rate,
            ),
        )

    def forward(self, x):
        return self.stconvs2s_c(x)


class STConvS2S_C_TemporalAttention(nn.Module):
    """STConvS2S-C with residual temporal self-attention per spatial cell."""

    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate,
                 step=5, output_channels=None, attention_heads=4):
        super(STConvS2S_C_TemporalAttention, self).__init__()
        input_length = input_size[2]

        self.stconvs2s_c = Model(
            TemporalCausalBlock, input_size, num_layers, hidden_dim, kernel_size, device,
            dropout_rate, step, output_channels,
            feature_adapter_factory=lambda channels: ResidualTemporalSelfAttentionBlock(
                channels, sequence_length=input_length, num_heads=attention_heads,
                dropout_rate=dropout_rate,
            ),
        )

    def forward(self, x):
        return self.stconvs2s_c(x)


class Model(nn.Module):
    def __init__(self, TemporalBlockInstance, input_size, num_layers, hidden_dim, kernel_size,
                 device, dropout_rate, step=5, output_channels=None,
                 feature_adapter_factory=None):
        super(Model, self).__init__()
        
        initial_in_channels = input_size[1]
        input_length = input_size[2]
        # Use output_channels if specified, otherwise default to input channels
        final_out_channels = output_channels if output_channels is not None else initial_in_channels
        
        temporal_block = TemporalBlockInstance(input_size, num_layers, kernel_size, in_channels=initial_in_channels, 
                                       out_channels=hidden_dim, dropout_rate=dropout_rate, step=step)

        spatial_block = SpatialBlock(num_layers, kernel_size, in_channels=hidden_dim, 
                                     out_channels=hidden_dim, dropout_rate=dropout_rate)
        
        if step > input_length:
            generator_block = TemporalGeneratorBlock(input_size, kernel_size, in_channels=hidden_dim, 
                                                     out_channels=hidden_dim, dropout_rate=dropout_rate, step=step)
                                     
            self.conv = nn.Sequential(temporal_block, spatial_block, generator_block)
        else:
            self.conv = nn.Sequential(temporal_block, spatial_block)
        
        padding = kernel_size // 2
        self.conv_final = nn.Conv3d(in_channels=hidden_dim, out_channels=final_out_channels, kernel_size=kernel_size, 
                                    padding=padding)

        # Constructed after all baseline modules so common parameters have identical
        # initialization under the same random seed.
        self.feature_adapter = (
            feature_adapter_factory(hidden_dim)
            if feature_adapter_factory is not None
            else nn.Identity()
        )
                                            
    def forward(self, x):
        for index, block in enumerate(self.conv):
            x = block(x)
            if index == 1:
                x = self.feature_adapter(x)
        return self.conv_final(x)
