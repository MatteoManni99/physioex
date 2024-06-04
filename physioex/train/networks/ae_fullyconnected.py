from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SleepAutoEncoderModule

module_config = dict()

def get_layer_list(input_dim, output_dim, n_hlayers, hlayer_sizes):
    layers = [nn.Linear(input_dim, hlayer_sizes[0]), nn.LeakyReLU()]
    for i in range(n_hlayers-1):
        layers.extend([nn.Linear(hlayer_sizes[i], hlayer_sizes[i+1]), nn.LeakyReLU()])
    layers.append(nn.Linear(hlayer_sizes[-1], output_dim))

    return layers

class AutoEncoderFullyConnected(SleepAutoEncoderModule):
    def __init__(self, module_config=module_config):
        super(AutoEncoderFullyConnected, self).__init__(Net(module_config), module_config)

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.encoder = Encoder(module_config)
        self.decoder = Decoder(module_config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Encoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.L = config["seq_len"]
        self.nchan = config["in_channels"]
        self.T = config["T"]
        self.F = config["F"]
        self.input_dim = self.nchan * self.T * self.F
        self.epoch_encode_dim = config["epoch_encode_dim"]
        self.output_dim = config["latent_dim"]

        self.epoch_encoder = nn.Sequential(*get_layer_list(self.input_dim, self.epoch_encode_dim,
                                                           1, [128]), nn.LeakyReLU())
        
        self.sequence_encoder = nn.Sequential(*get_layer_list(self.epoch_encode_dim * self.L, self.output_dim * self.L,
                                                              1, [128]), nn.LeakyReLU())

        self.lin_encode = nn.Linear(self.output_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        #print("Encoder input shape: ", x.shape)
        x = x.view(-1, self.nchan * self.T * self.F)
        #print("Encoder reshaped input shape: ", x.shape)
        x = self.epoch_encoder(x)
        #print("Encoder epoch encoded shape: ", x.shape)
        x = x.view(-1, self.L * self.epoch_encode_dim)
        #print("Encoder reshaped epoch encoded shape: ", x.shape)
        x = self.sequence_encoder(x)
        #print("Encoder sequence encoded shape: ", x.shape)
        x = x.view(-1, self.L, self.output_dim)
        #print("Encoder reshaped sequence encoded shape: ", x.shape)
        x = self.lin_encode(x)
        #print("Encoder lin_encode shape: ", x.shape)
        x = self.layer_norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.L = config["seq_len"]
        self.nchan = config["in_channels"]
        self.T = config["T"]
        self.F = config["F"]
        self.input_dim = config["latent_dim"]
        self.epoch_encode_dim = config["epoch_encode_dim"]
        self.output_dim = self.nchan * self.T * self.F

        self.sequence_decoder = nn.Sequential(*get_layer_list(self.input_dim * self.L, self.epoch_encode_dim * self.L,
                                                            1, [128]), nn.LeakyReLU())
        
        self.epoch_decoder = nn.Sequential(*get_layer_list(self.epoch_encode_dim, self.output_dim,
                                                          1, [128]))
        
        self.lin_decode = nn.Sequential(nn.Linear(self.input_dim, self.input_dim), nn.LeakyReLU())
        self.layer_norm = nn.LayerNorm([self.nchan, self.T, self.F])

    def forward(self, x):
        x = self.lin_decode(x)
        #print("Decoder lin_decode shape: ", x.shape)
        x = x.view(-1, self.L * self.input_dim)
        #print("Decoder reshaped lin_decode shape: ", x.shape)
        x = self.sequence_decoder(x)
        #print("Decoder sequence decoded shape: ", x.shape)
        x = x.view(-1, self.epoch_encode_dim)
        #print("Decoder reshaped sequence decoded shape: ", x.shape)
        x = self.epoch_decoder(x)
        #print("Decoder epoch decoded shape: ", x.shape)
        x = x.view(-1, self.L, self.nchan, self.T, self.F)
        #print("Decoder reshaped epoch decoded shape: ", x.shape)
        x = self.layer_norm(x)
        return x

    