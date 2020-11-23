import torch
import torch.nn as nn
import torch.nn.functional as F
from tool.utils import Util


class TemporalReversedBlock(nn.Module):
    def __init__(self, input_size, num_layers, kernel_size, in_channels, out_channels, dropout_rate, step):
        super(TemporalReversedBlock, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.conv_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()        
        self.input_length = input_size[2]
        self.step = step
        
        kernel_size = Util.generate_list_from(kernel_size)
        #factorized kernel
        temporal_kernel_size = [kernel_size[0], 1, 1]
        intermed_channels = out_channels
        
        for i in range(num_layers):
            intermed_channels*=2
            if i == (num_layers-1):
                intermed_channels = out_channels 
            self.conv_layers.append(
           	    RNet(in_channels, intermed_channels, kernel_size=temporal_kernel_size, bias=False)
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
        input_ = torch.flip(input_,[2])
        if self.dropout_rate > 0.:
            output = self.learning_with_dropout(input_)
        else:
            output = self.learning_without_dropout(input_)
            
        output = torch.flip(output,[2]) 
        return output


class RNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(RNet, self).__init__()
        
        self.temporal_kernel_value = kernel_size[0]
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
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
         



class TemporalCausalBlock(nn.Module):
    def __init__(self, input_size, num_layers, kernel_size, in_channels, out_channels, dropout_rate, step):
        super(TemporalCausalBlock, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.conv_layers = nn.ModuleList()
        self.lrelu_layers = nn.ModuleList()
        self.batch_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_length = input_size[2]
        self.step = step
        
        kernel_size = Util.generate_list_from(kernel_size)
        #factorized kernel
        temporal_kernel_size = [kernel_size[0], 1, 1]
        self.temporal_padding_value = kernel_size[0] - 1      
        temporal_padding = [self.temporal_padding_value, 0, 0]
        intermed_channels = out_channels
        for i in range(num_layers):
            intermed_channels*=2
            if i == (num_layers-1):
                intermed_channels = out_channels
            self.conv_layers.append(
                nn.Conv3d(in_channels, intermed_channels, kernel_size=temporal_kernel_size, 
                          padding=temporal_padding, bias=False)
            )
            self.lrelu_layers.append(nn.LeakyReLU())
            self.batch_layers.append(nn.BatchNorm3d(intermed_channels))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            in_channels = intermed_channels
        
        
    def learning_with_dropout(self, x):
        for conv, lrelu, batch, drop in zip(self.conv_layers, self.lrelu_layers, 
                                            self.batch_layers, self.dropout_layers):
            x = conv(x)[:,:,:-self.temporal_padding_value,:,:]
            x = drop(lrelu(batch(x)))
            
        return x
    
    def learning_without_dropout(self, x):
        for conv, lrelu, batch in zip(self.conv_layers, self.lrelu_layers, self.batch_layers):
            x = conv(x)[:,:,:-self.temporal_padding_value,:,:]
            x = lrelu(batch(x))
            
        return x
        
    def forward(self, input_):
        if self.dropout_rate > 0.:
            output = self.learning_with_dropout(input_)
        else:
            output = self.learning_without_dropout(input_)

        return output