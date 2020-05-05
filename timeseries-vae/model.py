import numpy as np
import torch
from torch import nn, optim


class Vrae(nn.Module):
    """VAE + LSTM"""
    def __init__(self, num_features, hidden_size, hidden_layer_depth, latent_length, sequence_length, output_size, batch_size, dropout):
        super(Vrae, self).__init__()
        # Parameters
        self.number_of_features = num_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.batch_size = batch_size
        
        # Encoder + reparameterization
        self.encoder = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout=dropout)
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)
            # Initialization
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)
        
        # Decoder
        self.decoder = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(torch.cuda.FloatTensor)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(torch.cuda.FloatTensor)
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)
            # Initialization
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)
    
    def get_latent(self, cell_output, training=True):
        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)
        
        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean
        
    
    def forward(self, x):
        # Encoding
#         print(type(x))
        _, (h_end, c_end) = self.encoder(x)
        h_end = h_end[-1, :, :]
        # Reparameterization
        latent = self.get_latent(h_end)
        # Decoding
        h_state = self.latent_to_hidden(latent)
            # repeat vector
        h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
        decoder_output, _ = self.decoder(self.decoder_inputs, (h_0, self.c_0))
        out = self.hidden_to_output(decoder_output)
        return out, self.latent_mean, self.latent_logvar  