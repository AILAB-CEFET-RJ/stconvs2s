import torch.nn as nn
import math

from tool.utils import Util


class Conv2Plus1D(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(Conv2Plus1D, self).__init__()
        self.conv2plus1_layers = nn.ModuleList()        
        initial_in_channels = input_size[1]
        in_channels = initial_in_channels
        out_channels = hidden_dim
        for i in range(num_layers):
            self.conv2plus1_layers.append(
                nn.Sequential(
                    Conv2Plus1Block(kernel_size, in_channels, out_channels, dropout_rate, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.LeakyReLU(inplace=True)
                ) 
            )
            in_channels = out_channels
        
        padding = kernel_size // 2
        self.out_conv = nn.Conv3d(in_channels=out_channels, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                  padding=padding)

        
    def forward(self, x):
        for conv2plus1 in self.conv2plus1_layers:
            x = conv2plus1(x)
                        
        return self.out_conv(x)    
        
            
        
class Conv2Plus1Block(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dropout_rate, bias):
        super(Conv2Plus1Block, self).__init__()
                  
        kernel_size = Util.generate_list_from(kernel_size)
        #factorized kernel
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        temporal_kernel_size = [kernel_size[0], 1, 1]
          
        spatial_padding_value = kernel_size[1] // 2
        temporal_padding_value = kernel_size[0] // 2      
        
        spatial_padding =  [0, spatial_padding_value, spatial_padding_value]
        temporal_padding = [temporal_padding_value, 0, 0]

        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
    
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size, 
                      padding=spatial_padding, bias=bias),
            nn.BatchNorm3d(intermed_channels),
            nn.LeakyReLU(inplace=True)                                            
        )
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                       padding=temporal_padding, bias=bias)
         
        
    def forward(self, x):
        x = self.spatial_conv(x)
        return self.temporal_conv(x)



"""
class R2Plus1(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(R2Plus1, self).__init__()
        self.r2plus1_layers = nn.ModuleList()
        self.lrelu_layers = nn.ModuleList()
        self.batch_layers = nn.ModuleList()
        
        initial_in_channels = input_size[1]
        in_channels = initial_in_channels
        out_channels = hidden_dim
        for i in range(num_layers):
            self.r2plus1_layers.append(
                R2Plus1Block(kernel_size, in_channels, out_channels, dropout_rate)
            )
            self.batch_layers.append(nn.BatchNorm3d(out_channels))
            self.lrelu_layers.append(nn.LeakyReLU())
            in_channels = out_channels
        
        padding = kernel_size // 2
        self.out_conv = nn.Conv3d(in_channels=out_channels, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                  padding=padding)

        
    def forward(self, x):
        for r2plus1, lrelu, batch in zip(self.r2plus1_layers, self.lrelu_layers, self.batch_layers):
            x = lrelu(batch(r2plus1(x)))
                        
        return self.out_conv(x)    
        
            
class R2Plus1CausalBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dropout_rate):
        super(R2Plus1CausalBlock, self).__init__()
                  
        kernel_size = Util.generate_list_from(kernel_size)
        #factorized kernel
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        temporal_kernel_size = [kernel_size[0], 1, 1]
          
        spatial_padding_value = kernel_size[1] // 2
        self.temporal_padding_value = kernel_size[0] - 1      
        
        spatial_padding =  [0, spatial_padding_value, spatial_padding_value]
        temporal_padding = [self.temporal_padding_value, 0, 0]
        # number of parameters in the (2+1)D block is approximately equal to that implementing full 3D convolution.
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
    
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size, 
                      padding=spatial_padding, bias=False),
            nn.BatchNorm3d(intermed_channels),
            nn.LeakyReLU(inplace=True)                                            
        )
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                       padding=temporal_padding, bias=False)
         
        
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)[:,:,:-self.temporal_padding_value,:,:]
        
        return x
        
        
class R2Plus1Block(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dropout_rate):
        super(R2Plus1Block, self).__init__()
                  
        kernel_size = Util.generate_list_from(kernel_size)
        #factorized kernel
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        temporal_kernel_size = [kernel_size[0], 1, 1]
          
        spatial_padding_value = kernel_size[1] // 2
        temporal_padding_value = kernel_size[0] // 2      
        
        spatial_padding =  [0, spatial_padding_value, spatial_padding_value]
        temporal_padding = [temporal_padding_value, 0, 0]
        # number of parameters in the (2+1)D block is approximately equal to that implementing full 3D convolution.
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
    
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size, 
                      padding=spatial_padding, bias=False),
            nn.BatchNorm3d(intermed_channels),
            nn.LeakyReLU(inplace=True)                                            
        )
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                       padding=temporal_padding, bias=False)
         
        
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        
        return x
        
        
 """       