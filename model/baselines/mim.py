#-----------------------------------------------------------
# my implementation based on https://github.com/Yunbo426/MIM 
#-----------------------------------------------------------

import torch
import torch.nn as nn
from tool.utils import Util

class MIM(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(MIM, self).__init__()

        self.filter_size = kernel_size
        self.num_hidden_out = input_size[1]
        self.input_length = input_size[2]
        self.step = step
        self.num_layers = num_layers
        self.num_hidden = Util.generate_list_from(hidden_dim,num_layers)
        self.device = device
        self.stlstm_layer = nn.ModuleList()
        self.stlstm_layer_diff = nn.ModuleList()
        
        num_hidden_in = self.num_hidden_out
        for i in range(self.num_layers):
            if i < 1:
                self.stlstm_layer.append(
                    SpatioTemporalLSTMCell(self.filter_size, num_hidden_in, self.num_hidden[i], input_size, device, dropout_rate)
                )
            else:
                self.stlstm_layer.append(
                    MIMBlock(self.filter_size, self.num_hidden[i], input_size, device, dropout_rate)
                )
                     
        for i in range(self.num_layers - 1):
            self.stlstm_layer_diff.append(
                MIMS(self.filter_size, self.num_hidden[i+1], input_size, device, dropout_rate)
            )

        self.conv_last = nn.Conv2d(self.num_hidden[num_layers - 1], self.num_hidden_out,
                                   kernel_size=1, stride=1, padding=0, bias=False) 
                                   
        
    def forward(self, frames):
        #[batch, channel, length, height, width] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        schedual_sampling_bool = torch.zeros((batch, self.step - self.input_length,
                                              frames.shape[2], height, width)).to(self.device)
        convlstm_c = None
        
        next_frames = []
        hidden_state, cell_state = [],[]
        hidden_state_diff, cell_state_diff = [],[]

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.device)
            hidden_state.append(zeros)
            cell_state.append(zeros)
            if i < (self.num_layers - 1):
                hidden_state_diff.append(zeros)
                cell_state_diff.append(zeros)

        st_memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.device)
        for time_step in range(self.step):
            if time_step < self.input_length:
                x_gen = frames[:,time_step]
            else:
                x_gen = (1 - schedual_sampling_bool[:,time_step - self.input_length]) * x_gen
                
            preh = hidden_state[0]
            hidden_state[0], cell_state[0], st_memory = self.stlstm_layer[0](x_gen, hidden_state[0], cell_state[0], st_memory)

            for i in range(1, self.num_layers):
                if time_step > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                             
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                
                else:
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(hidden_state[i - 1]), None, None)
            
                preh = hidden_state[i]
                
                hidden_state[i], cell_state[i], st_memory, convlstm_c = self.stlstm_layer[i](
                    hidden_state[i - 1], hidden_state_diff[i - 1], hidden_state[i], cell_state[i], st_memory, convlstm_c)   
                
            x_gen = self.conv_last(hidden_state[self.num_layers-1])
            next_frames.append(x_gen)


        # [length, batch, channel, height, width] -> [batch, channel, length, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 2, 0, 3, 4).contiguous()
        
        return next_frames



class MIMBlock(nn.Module):
    def __init__(self, filter_size, num_hidden, input_size, device, dropout_rate, bias=False):
        super(MIMBlock, self).__init__()
        
        self.num_hidden = num_hidden
        self.width = input_size[4]
        self.device = device
        self._forget_bias = 1.0

        pad = filter_size // 2
        self.t_cc = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=1, padding=pad, bias=bias),
            nn.LayerNorm([num_hidden * 3, self.width, self.width]),
            nn.Dropout2d(dropout_rate)
        )        
        self.s_cc = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=pad, bias=bias),
            nn.LayerNorm([num_hidden * 4, self.width, self.width]),
            nn.Dropout2d(dropout_rate)
        )        
        self.x_cc = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=pad, bias=bias),
            nn.LayerNorm([num_hidden * 4, self.width, self.width]),
            nn.Dropout2d(dropout_rate)
        ) 
        self.mims = MIMS(filter_size, num_hidden, input_size, device, dropout_rate)
        self.last = nn.Conv2d(num_hidden * 2, 1, 1, 1, 0) 

    def init_state(self, shape):
        return torch.zeros([shape[0], self.num_hidden, shape[2], shape[3]]).to(self.device)
        
    def forward(self, x, diff_h, h, c, m, convlstm_c):
        if h is None:
            h = self.init_state(x.shape)
        if c is None:
            c = self.init_state(x.shape)
        if m is None:
            m = self.init_state(x.shape)
        if diff_h is None:
            diff_h = torch.zeros_like(h)
            
        i_s, g_s, f_s, o_s = torch.split(self.s_cc(m), self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(self.t_cc(h), self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(self.x_cc(x), self.num_hidden, dim=1)
                 
        i  = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g  = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o  = torch.sigmoid(o_x + o_t + o_s)
        
        new_m = f_ * m + i_ * g_
        c, new_convlstm_c = self.mims(diff_h, c, convlstm_c)
        new_c = c + i * g
        cell = self.last(torch.cat([new_c, new_m], 1))
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m, new_convlstm_c
        

class MIMS(nn.Module):
    def __init__(self, filter_size, num_hidden, input_size, device, dropout_rate, bias=False):
        super(MIMS, self).__init__()
        
        self.num_hidden = num_hidden
        self.width = input_size[4]
        self.device = device
        self._forget_bias = 1.0

        pad = filter_size // 2                           
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=pad, bias=bias),
            nn.LayerNorm([num_hidden * 4, self.width, self.width]),
            nn.Dropout2d(dropout_rate)
        )
        self.conv_x = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=pad, bias=bias),
            nn.LayerNorm([num_hidden * 4, self.width, self.width]),
            nn.Dropout2d(dropout_rate)
        )         
        self.ct_weight = nn.init.normal_(nn.Parameter(torch.Tensor(torch.zeros(num_hidden*2, self.width, self.width))), 0.0, 1.0)
        self.oc_weight = nn.init.normal_(nn.Parameter(torch.Tensor(torch.zeros(num_hidden, self.width, self.width))), 0.0, 1.0)
        
    def init_state(self, shape):
        out = torch.zeros([shape[0], self.num_hidden, shape[2], shape[3]]).to(self.device)
        return out 
        
    def forward(self, x, h_t, c_t):
        if h_t is None:
            h_t = self.init_state(x.shape)
        if c_t is None:
            c_t = self.init_state(x.shape)
        
        out = self.conv_h(h_t) 
        i_h, g_h, f_h, o_h = torch.split(out, self.num_hidden, dim=1)
        ct_activation = torch.matmul(c_t.repeat([1,2,1,1]), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h
        
        if x is not None:
            outx = self.conv_x(x) 
            i_x, g_x, f_x, o_x = torch.split(outx, self.num_hidden, dim=1)
            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.matmul(c_new, self.oc_weight)
        
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new
        
        
class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, filter_size, num_hidden_in, num_hidden, input_size, device, dropout_rate, bias=False):
        super(SpatioTemporalLSTMCell, self).__init__()
        
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.width = input_size[4]
        self.device = device
        self._forget_bias = 1.0
        
        pad = filter_size // 2
        self.t_cc = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=pad, bias=bias),
            nn.LayerNorm([num_hidden * 4, self.width, self.width]),
            nn.Dropout2d(dropout_rate)
        )        
        self.s_cc = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=pad, bias=bias),
            nn.LayerNorm([num_hidden * 4, self.width, self.width]),
            nn.Dropout2d(dropout_rate)
        )        
        self.x_cc = nn.Sequential(
           nn.Conv2d(num_hidden_in, num_hidden * 4, kernel_size=filter_size, stride=1, padding=pad, bias=bias),
           nn.LayerNorm([num_hidden * 4, self.width, self.width]),
           nn.Dropout2d(dropout_rate)
        )  
        self.last = nn.Conv2d(num_hidden * 2, 1, 1, 1, 0) 
        
    def init_state(self, shape):
        return torch.zeros([shape[0], self.num_hidden, shape[2], shape[3]]).to(self.device)
    
    def forward(self, x, h, c, m):
        if h is None:
            h = self.init_state(x.shape)
        if c is None:
            c = self.init_state(x.shape)
        if m is None:
            m = self.init_state(x.shape)
            
        i_s, g_s, f_s, o_s = torch.split(self.s_cc(m), self.num_hidden, dim=1)
        i_t, g_t, f_t, o_t = torch.split(self.t_cc(h), self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(self.x_cc(x), self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.sigmoid(g_x + g_t)
        g_ = torch.sigmoid(g_x + g_s)
        f = torch.sigmoid(f_x + f_t + self._forget_bias)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        
        new_m = f_ * m + i_ * g_
        new_c = f * c + i * g
        cell = self.last(torch.cat([new_c, new_m], 1))
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m