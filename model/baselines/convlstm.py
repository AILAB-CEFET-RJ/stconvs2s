#------------------------------------------------------------------------- 
# implementation based on https://github.com/AlexMa011/ConvLSTM_pytorch 
#-------------------------------------------------------------------------

import torch.nn as nn
import torch

class STConvLSTM(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, kernel_size, device, dropout_rate, step=5):
        super(STConvLSTM, self).__init__()
                
        input_dim = input_size[1]
        output_dim = input_dim
        self.convlstm = ConvLSTM(input_size, input_dim, hidden_dim, (kernel_size,kernel_size), 
                                 num_layers, dropout_rate, step, device)

        self.conv_layer = nn.Conv3d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=1, padding=0)
        
    def forward(self, input_):
        output = self.convlstm(input_)
        return self.conv_layer(output)
        
    
class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, 
                 dropout_rate, step, device, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.device = device
        self.input_length = input_size[2]
        self.height = input_size[3]
        self.width = input_size[4]
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.step = step

        cell_list, conv_list = [],[]
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias, dropout_rate=dropout_rate,
                                          device=self.device
                                         ))
            conv_list.append(nn.Conv2d(self.hidden_dim[num_layers-1], cur_input_dim,
                                       kernel_size=1, stride=1, padding=0, bias=False))

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_list = nn.ModuleList(conv_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (c, b, t, h, w) -> (b, c, t, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)
      
        batch = input_tensor.shape[0]
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=batch)
           
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []            
            for t in range(self.step):
                if t < self.input_length:
                    net = cur_layer_input[:, :, t, :, :]
                else:
                    net = x_gen
                
                h, c = self.cell_list[layer_idx](input_tensor=net, cur_state=[h, c])
                output_inner.append(h)
                x_gen = self.conv_list[layer_idx](h)

            layer_output = torch.stack(output_inner, dim=2)
            cur_layer_input = layer_output

        return layer_output

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
        
        
class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dropout_rate, device):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
                    
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      bias=self.bias),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(self.device)
        c = h
        return (h, c)