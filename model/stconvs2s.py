import torch
import torch.nn as nn

from .temporal_block import TemporalReversedBlock, TemporalCausalBlock
from .spatial_block import SpatialBlock
from .generator_block import TemporalGeneratorBlock


class STConvS2S_R(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(STConvS2S_R, self).__init__()
        
        self.stconvs2s_r = Model(TemporalReversedBlock, input_size, num_layers, hidden_dim, 
                                 kernel_size, device, dropout_rate, step)
                                            
    def forward(self, x):
        return self.stconvs2s_r(x)

        
     
class STConvS2S_C(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(STConvS2S_C, self).__init__()
        
        initial_in_channels = input_size[1]
        input_length = input_size[2]
        
        self.stconvs2s_c = Model(TemporalCausalBlock, input_size, num_layers, hidden_dim, 
                                 kernel_size, device, dropout_rate, step)
                                            
    def forward(self, x):
        return self.stconvs2s_c(x)        


class Model(nn.Module):
    def __init__(self, TemporalBlockInstance, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(Model, self).__init__()
        
        initial_in_channels = input_size[1]
        input_length = input_size[2]
        
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
        self.conv_final = nn.Conv3d(in_channels=hidden_dim, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                    padding=padding)
                                            
    def forward(self, x):
        x = self.conv(x)
        return self.conv_final(x)        