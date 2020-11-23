import torch
import torch.nn as nn

from .temporal_block import *
from .spatial_block import *


class AblationSTConvS2S_R_Inverted(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(AblationSTConvS2S_R_Inverted, self).__init__()
        
        self.stconvs2s_r = ModelInverted(TemporalReversedBlock, SpatialBlock, input_size, 
                                         num_layers, hidden_dim, kernel_size, device, dropout_rate, step)
                                            
    def forward(self, x):
        return self.stconvs2s_r(x)



class AblationSTConvS2S_C_Inverted(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(AblationSTConvS2S_C_Inverted, self).__init__()
        
        self.stconvs2s_c = ModelInverted(TemporalCausalBlock, SpatialBlock, input_size, 
                                         num_layers, hidden_dim, kernel_size, device, dropout_rate, step)
                                            
    def forward(self, x):
        return self.stconvs2s_c(x)
        
        

class AblationSTConvS2S_R_NoChannelIncrease(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(AblationSTConvS2S_R_NoChannelIncrease, self).__init__()
        
        self.stconvs2s_r = Model(TemporalReversedBlock_NoChannelIncrease, SpatialBlock_NoChannelIncrease, input_size, 
                                 num_layers, hidden_dim, kernel_size, device, dropout_rate, step)
                                            
    def forward(self, x):
        return self.stconvs2s_r(x)

        
     
class AblationSTConvS2S_C_NoChannelIncrease(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step):
        super(AblationSTConvS2S_C_NoChannelIncrease, self).__init__()
        
        initial_in_channels = input_size[1]
        input_length = input_size[2]
        
        self.stconvs2s_c = Model(TemporalCausalBlock_NoChannelIncrease, SpatialBlock_NoChannelIncrease, input_size, 
                                 num_layers, hidden_dim, kernel_size, device, dropout_rate, step)
                                            
    def forward(self, x):
        return self.stconvs2s_c(x)
        


class AblationSTConvS2S_NoCausalConstraint(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step):
        super(AblationSTConvS2S_NoCausalConstraint, self).__init__()
        
        initial_in_channels = input_size[1]
        input_length = input_size[2]
        
        self.stconvs2s = Model(TemporalBlock, SpatialBlock, input_size, num_layers, hidden_dim, 
                               kernel_size, device, dropout_rate, step)
                                            
    def forward(self, x):
        return self.stconvs2s(x)
        


class AblationSTConvS2S_NoTemporal(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step):
        super(AblationSTConvS2S_NoTemporal, self).__init__()
        
        initial_in_channels = input_size[1]
        input_length = input_size[2]
        
        self.conv = SpatialBlock(num_layers, kernel_size, in_channels=initial_in_channels, 
                                     out_channels=hidden_dim, dropout_rate=dropout_rate)
        
        padding = kernel_size // 2
        self.conv_final = nn.Conv3d(in_channels=hidden_dim, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                    padding=padding)
                                            
    def forward(self, x):
        x = self.conv(x)
        return self.conv_final(x)
          
        
        
class AblationSTConvS2S_R_NotFactorized(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(AblationSTConvS2S_R_NotFactorized, self).__init__()
        
        self.conv3D_layers = nn.ModuleList()
        initial_in_channels = input_size[1]
        in_channels = initial_in_channels
        out_channels = hidden_dim   
        kernel_size = Util.generate_list_from(kernel_size)
        spatial_padding_value = kernel_size[1] // 2
        temporal_padding_value = kernel_size[0] // 2     
        padding = [temporal_padding_value, spatial_padding_value, spatial_padding_value]

        for i in range(num_layers):
            self.conv3D_layers.append(
                RNetNotFactorized(in_channels, out_channels, kernel_size, bias=False)
            )
            in_channels = out_channels
        
        self.conv_final = nn.Conv3d(in_channels=hidden_dim, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                    padding=padding)
        
    def forward(self, x):
        x = torch.flip(x,[2])
        for conv in self.conv3D_layers:
            x = conv(x)
        
        output = torch.flip(x,[2])                
        return self.conv_final(output)        

        
     
class AblationSTConvS2S_C_NotFactorized(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step):
        super(AblationSTConvS2S_C_NotFactorized, self).__init__()

        self.conv3D_layers = nn.ModuleList()
        initial_in_channels = input_size[1]
        in_channels = initial_in_channels
        out_channels = hidden_dim      
        for i in range(num_layers):
            self.conv3D_layers.append(
                nn.Sequential(
                    Conv3DCausalBlock(kernel_size, in_channels, out_channels, dropout_rate, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.LeakyReLU(inplace=True)
                ) 
            )
            in_channels = out_channels
        
        padding = kernel_size // 2
        self.conv_final = nn.Conv3d(in_channels=hidden_dim, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                    padding=padding)
   
    def forward(self, x):
        for conv in self.conv3D_layers:
            x = conv(x)
                        
        return self.conv_final(x)  
        
    

class Model(nn.Module):
    def __init__(self, TemporalBlockInstance, SpatialBlockInstance, input_size, num_layers, hidden_dim, 
                 kernel_size, device, dropout_rate, step=5):
        super(Model, self).__init__()
        
        initial_in_channels = input_size[1]
        input_length = input_size[2]
        
        temporal_block = TemporalBlockInstance(input_size, num_layers, kernel_size, in_channels=initial_in_channels, 
                                               out_channels=hidden_dim, dropout_rate=dropout_rate, step=step)
        
        spatial_block = SpatialBlockInstance(num_layers, kernel_size, in_channels=hidden_dim, 
                                             out_channels=hidden_dim, dropout_rate=dropout_rate)
                                     
        self.conv = nn.Sequential(temporal_block, spatial_block)
        
        padding = kernel_size // 2
        self.conv_final = nn.Conv3d(in_channels=hidden_dim, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                    padding=padding)
                                            
    def forward(self, x):
        x = self.conv(x)
        return self.conv_final(x)       
        
        
        
class ModelInverted(nn.Module):
    def __init__(self, TemporalBlockInstance, SpatialBlockInstance, input_size, num_layers, hidden_dim, 
                 kernel_size, device, dropout_rate, step=5):
        super(ModelInverted, self).__init__()
        
        initial_in_channels = input_size[1]
        input_length = input_size[2]
        
        spatial_block = SpatialBlockInstance(num_layers, kernel_size, in_channels=initial_in_channels, 
                                             out_channels=hidden_dim, dropout_rate=dropout_rate)
                                             
        temporal_block = TemporalBlockInstance(input_size, num_layers, kernel_size, in_channels=hidden_dim, 
                                               out_channels=hidden_dim, dropout_rate=dropout_rate, step=step)
                                     
        self.conv = nn.Sequential(spatial_block, temporal_block)
        
        padding = kernel_size // 2
        self.conv_final = nn.Conv3d(in_channels=hidden_dim, out_channels=initial_in_channels, kernel_size=kernel_size, 
                                    padding=padding)
                                            
    def forward(self, x):
        x = self.conv(x)
        return self.conv_final(x)  
        
 
 
class Conv3DCausalBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dropout_rate, bias):
        super(Conv3DCausalBlock, self).__init__()
                  
        kernel_size = Util.generate_list_from(kernel_size)
        spatial_padding_value = kernel_size[1] // 2
        self.temporal_padding_value = kernel_size[0] - 1      
        
        padding =  [self.temporal_padding_value, spatial_padding_value, spatial_padding_value]
    
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
                 
    def forward(self, x):
        return self.conv(x)[:,:,:-self.temporal_padding_value,:,:]
        
        

class RNetNotFactorized(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(RNetNotFactorized, self).__init__()
        
        self.temporal_kernel_value = kernel_size[0]
        spatial_padding_value = kernel_size[1] // 2
        padding =  [0, spatial_padding_value, spatial_padding_value]
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_k2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=[2,1,1], bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.pad_k2 = nn.ReplicationPad3d((0, 0, 0, 0, 0, 1))
            
    def forward(self, x):
        if self.temporal_kernel_value == 2:
            return self.conv_k2(x)
            
        output_conv = self.conv(x)
        output_conv_k2 = self.conv_k2(x[:,:,-self.temporal_kernel_value:,:,:])
        output_conv_k2 = self.pad_k2(output_conv_k2)
        output_conv_part_1 = output_conv[:,:,:-1,:,:]
        output_conv_part_2 = output_conv[:,:,-1:,:,:]
        output_conv_part_2 =  output_conv_part_2 - output_conv_k2
        output = torch.cat([output_conv_part_1,output_conv_part_2], dim=2)  
        return output  