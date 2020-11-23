import torch.nn as nn
from tool.utils import Util


class SpatialBlock(nn.Module):
    def __init__(self, num_layers, kernel_size, in_channels, out_channels, dropout_rate):
        super(SpatialBlock, self).__init__()
        self.padding = kernel_size // 2
        self.dropout_rate = dropout_rate
        self.conv_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        kernel_size = Util.generate_list_from(kernel_size)
        #factorized kernel
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]          
        spatial_padding_value = kernel_size[1] // 2        
        spatial_padding =  [0, spatial_padding_value, spatial_padding_value]
        intermed_channels = out_channels
        for i in range(num_layers):
            intermed_channels*=2
            if i == (num_layers-1):
                intermed_channels = out_channels 
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, intermed_channels, kernel_size=spatial_kernel_size, 
                              padding=spatial_padding, bias=False),
                    nn.BatchNorm3d(intermed_channels),
                    nn.LeakyReLU(inplace=True)
                )            
            )
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            in_channels = intermed_channels
        
            
    def learning_with_dropout(self, x):
        for conv, drop in zip(self.conv_layers, self.dropout_layers):
            x = drop(conv(x))
            
        return x
    
    def learning_without_dropout(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            
        return x
        
    def forward(self, input_):
        if self.dropout_rate > 0.:
            output = self.learning_with_dropout(input_)
        else:
            output = self.learning_without_dropout(input_)
        
        return output