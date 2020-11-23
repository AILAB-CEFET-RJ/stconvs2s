import math
import torch
import torch.nn as nn
from tool.utils import Util


class TemporalGeneratorBlock(nn.Module):
    def __init__(self, input_size, kernel_size, in_channels, out_channels, dropout_rate, step):
        super(TemporalGeneratorBlock, self).__init__()  
        self.step = step
        self.input_length = input_size[2]
        self.tconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        kernel_size = Util.generate_list_from(kernel_size)
        #factorized kernel
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]          
        spatial_padding_value = kernel_size[1] // 2        
        spatial_padding =  [0, spatial_padding_value, spatial_padding_value]
        
        num_layers = math.ceil((self.step - self.input_length)/(2 * self.input_length))
        intermed_channels = out_channels
        for i in range(num_layers):
            intermed_channels*=2
            if i == (num_layers-1):
                intermed_channels = out_channels
            self.tconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_channels, intermed_channels, [4,1,1], 
                                       stride=[2,1,1], padding=[1,0,0], bias=False),
                    nn.BatchNorm3d(intermed_channels),
                    nn.LeakyReLU(inplace=True)
                )            
            )
            in_channels = intermed_channels
            
        num_layers = self.step // self.input_length
        intermed_channels*=2
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, intermed_channels, kernel_size=spatial_kernel_size, 
                              padding=spatial_padding, bias=False),
                    nn.BatchNorm3d(intermed_channels),
                    nn.LeakyReLU(inplace=True)
                )
            )
            in_channels = intermed_channels
            intermed_channels = out_channels

                                                
    def crop(self, tensor, target_depth):
        diff_z = (tensor.shape[2] - target_depth) // 2
        return tensor[:, :, diff_z : (diff_z + target_depth), :,:]
        
    def forward(self, input_):    
        x = input_.clone()
        for tconv in self.tconv_layers:
            x = tconv(x)
               
        output = torch.cat([input_, x], dim=2)
        if output.shape[2] > self.step:
            output = self.crop(output, self.step)
        
        for conv in self.conv_layers:
            output = conv(output)
            
        return output

"""

class TemporalGeneratorBlock(nn.Module):
    def __init__(self, input_size, kernel_size, in_channels, out_channels, dropout_rate, step):
        super(TemporalGeneratorBlock, self).__init__()  
        self.step = step
        self.input_length = input_size[2]
        self.tconv_layers = nn.ModuleList()
        num_layers = math.ceil((self.step - self.input_length)/(2 * self.input_length))
        intermed_channels = out_channels
        for i in range(num_layers):
            intermed_channels*=2
            if i == (num_layers-1):
                intermed_channels = out_channels
            self.tconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_channels, intermed_channels, [4,1,1], 
                                       stride=[2,1,1], padding=[1,0,0], bias=False),
                    nn.BatchNorm3d(intermed_channels),
                    nn.LeakyReLU(inplace=True)
                )            
            )
            in_channels = intermed_channels

        #num_layers = self.step // self.input_length
        #self.conv = SpatialBlock(num_layers, kernel_size, intermed_channels, out_channels, dropout_rate)           

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=[1, kernel_size, kernel_size], 
                          padding=[0, kernel_size // 2, kernel_size // 2], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels*2, kernel_size=[1, kernel_size, kernel_size], 
                          padding=[0, kernel_size // 2, kernel_size // 2], bias=False),
            nn.BatchNorm3d(out_channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels*2, out_channels, kernel_size=[1, kernel_size, kernel_size], 
                          padding=[0, kernel_size // 2, kernel_size // 2], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        melhor por enquanto - 6.2590 e 2.8054
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels*2, kernel_size=[1, kernel_size, kernel_size], 
                          padding=[0, kernel_size // 2, kernel_size // 2], bias=False),
            nn.BatchNorm3d(out_channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels*2, out_channels, kernel_size=[1, kernel_size, kernel_size], 
                          padding=[0, kernel_size // 2, kernel_size // 2], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=[1, kernel_size, kernel_size], 
                          padding=[0, kernel_size // 2, kernel_size // 2], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels*2, kernel_size=[1, kernel_size, kernel_size], 
                          padding=[0, kernel_size // 2, kernel_size // 2], bias=False),
            nn.BatchNorm3d(out_channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels*2, out_channels*2, kernel_size=[1, kernel_size, kernel_size], 
                          padding=[0, kernel_size // 2, kernel_size // 2], bias=False),
            nn.BatchNorm3d(out_channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels*2, out_channels, kernel_size=[1, kernel_size, kernel_size], 
                          padding=[0, kernel_size // 2, kernel_size // 2], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        ) 
                                                
    def crop(self, tensor, target_depth):
        diff_z = (tensor.shape[2] - target_depth) // 2
        return tensor[:, :, diff_z : (diff_z + target_depth), :,:]
        
    def forward(self, input_):    
        x = input_.clone()
        for tconv in self.tconv_layers:
            x = tconv(x)
               
        output = torch.cat([input_, x], dim=2)
        if output.shape[2] > self.step:
            output = self.crop(output, self.step)

        return self.conv(output)


"""