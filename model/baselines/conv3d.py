import torch.nn as nn
import math

from tool.utils import Util


class Conv3D(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(Conv3D, self).__init__()
        self.conv3D_layers = nn.ModuleList()
        initial_in_channels = input_size[1]
        in_channels = initial_in_channels
        out_channels = hidden_dim      
        for i in range(num_layers):
            self.conv3D_layers.append(
                nn.Sequential(
                    Conv3DBlock(kernel_size, in_channels, out_channels, dropout_rate, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.LeakyReLU(inplace=True)
                ) 
            )
            in_channels = out_channels
        
        padding = kernel_size // 2
        self.conv_final = nn.Conv3d(in_channels=hidden_dim, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                    padding=padding)

        
    def forward(self, x):
        for conv3D in self.conv3D_layers:
            x = conv3D(x)
                        
        return self.conv_final(x)    
                
        
        
class Conv3DBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dropout_rate, bias):
        super(Conv3DBlock, self).__init__()
                  
        kernel_size = Util.generate_list_from(kernel_size)
        spatial_padding_value = kernel_size[1] // 2
        temporal_padding_value = kernel_size[0] // 2     
        
        padding =  [temporal_padding_value, spatial_padding_value, spatial_padding_value]
    
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
                 
    def forward(self, x):
        return self.conv(x)
        

"""
class C3D(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(C3D, self).__init__()
        self.c3d_layers = nn.ModuleList()
        self.lrelu_layers = nn.ModuleList()
        self.batch_layers = nn.ModuleList()
        
        initial_in_channels = input_size[1]
        in_channels = initial_in_channels
        out_channels = hidden_dim
        for i in range(num_layers):
            self.c3d_layers.append(
                C3DBlock(kernel_size, in_channels, out_channels, dropout_rate)
            )
            self.batch_layers.append(nn.BatchNorm3d(out_channels))
            self.lrelu_layers.append(nn.LeakyReLU())
            in_channels = out_channels
        
        padding = kernel_size // 2
        self.conv_final = nn.Conv3d(in_channels=hidden_dim, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                    padding=padding)

        
    def forward(self, x):
        for c3d, lrelu, batch in zip(self.c3d_layers, self.lrelu_layers, self.batch_layers):
            x = lrelu(batch(c3d(x)))
                        
        return self.conv_final(x)    
""" 
