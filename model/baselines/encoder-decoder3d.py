#----------------------------------------------------------------
# my implementation based on https://github.com/eracah/hur-detect
#----------------------------------------------------------------


import torch
import torch.nn as nn
from tool.utils import Util


class Endocer_Decoder3D(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(Endocer_Decoder3D, self).__init__()
        
        self.down_block = nn.ModuleList()
        self.up_block = nn.ModuleList()
        self.input_size = input_size
        initial_in_channels = input_size[1]
        in_channels = initial_in_channels
        out_channels = hidden_dim
        
        for i in range(num_layers):
            self.down_block.append(
                DownsampleBlock(kernel_size, in_channels, out_channels)
            )
            in_channels = out_channels
            out_channels = out_channels * 2
            
        out_channels = out_channels // 2
        self.batch = nn.BatchNorm3d(out_channels)

        for i in range(num_layers):
            in_channels = out_channels
            out_channels = int(out_channels /2)
            self.up_block.append(
                UpsampleBlock(kernel_size, in_channels, out_channels)
            )
        
        padding = kernel_size // 2
        self.out_conv = nn.Conv3d(in_channels=out_channels, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                  padding=padding)
        
    def crop(self, tensor, target_size):
        _, _, _, tensor_height, tensor_width = tensor.size()
        diff_y = (tensor_height - target_size[0]) // 2
        diff_x = (tensor_width - target_size[1]) // 2
        return tensor[:, :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]
            
    def forward(self, x):
        for down in self.down_block:
            x = down(x)

        x = self.batch(x)
        for up in self.up_block:
            x = up(x)
        
        out = self.out_conv(x)
        if out.shape[3:] != self.input_size[3:]:
            out = self.crop(out, self.input_size[3:])
        return out
        

class DownsampleBlock(nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        kernel_size = Util.generate_list_from(kernel_size)
        temporal_value = kernel_size[0] // 2
        spatial_value = kernel_size[1] // 2
        padding = [temporal_value, spatial_value, spatial_value]
        stride = [1, spatial_value, spatial_value]
        self.down_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        out = self.down_block(x)
        return out


class UpsampleBlock(nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        kernel_size = Util.generate_list_from(kernel_size)
        spatial_value = kernel_size[1] // 2
        kernel_size = [1, spatial_value, spatial_value]
        stride = [1, spatial_value, spatial_value]
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(inplace=True)                                             
        )

    def forward(self, x):
        out = self.conv(x)
        return out        