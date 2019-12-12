import torch.nn as nn

from .encoder import EncoderSTCNN
from .decoder import DecoderSTCNN

class STConvS2S(nn.Module):
    def __init__(self, channels, dropout_rate, upsample,
                  encoder_layer_size=3, decoder_layer_size=3, kernel_size=5, filter_size=32):
        super(STConvS2S, self).__init__()
        
        self.encoder = EncoderSTCNN(layer_size=encoder_layer_size, kernel_size=kernel_size, 
                                    initial_filter_size=filter_size, channels=channels, 
                                    dropout_rate=dropout_rate)
                                    
        self.decoder = DecoderSTCNN(layer_size=decoder_layer_size, kernel_size=kernel_size, 
                                    initial_filter_size=filter_size, channels=filter_size,
                                    dropout_rate=dropout_rate, upsample=upsample)
        
    def forward(self, x):
        out = self.encoder(x)
        return self.decoder(out)