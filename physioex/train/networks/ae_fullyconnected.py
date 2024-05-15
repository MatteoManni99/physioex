from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SleepAutoEncoderModule

module_config = dict()


class AutoEncoderFullyConnected(SleepAutoEncoderModule):
    def __init__(self, module_config=module_config):
        super(AutoEncoderFullyConnected, self).__init__(Net(module_config), module_config)

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.encoder = Encoder(module_config)
        self.decoder = Decoder(module_config)

    def forward(self, x):
        print (x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Encoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.act_fn = nn.ReLU()
        self.input_dim = config["input_dim"]
        self.output_dim = config["latent_dim"]
        self.n_hlayers = config["n_encoder_hlayers"]
        self.hlayer_sizes = config["encoder_hlayer_sizes"]

        layers = [nn.Linear(self.input_dim, self.hlayer_sizes[0]), self.act_fn]
        for i in range(self.n_hlayers-1):
            layers.extend([nn.Linear(self.hlayer_sizes[i], self.hlayer_sizes[i+1]), self.act_fn])
        layers.append(nn.Linear(self.hlayer_sizes[-1], self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.act_fn = nn.ReLU()
        self.input_dim = config["latent_dim"]
        self.output_dim = config["input_dim"]
        self.n_hlayers = config["n_decoder_hlayers"]
        self.hlayer_sizes = config["decoder_hlayer_sizes"]

        layers = [nn.Linear(self.input_dim, self.hlayer_sizes[0]), self.act_fn]
        for i in range(self.n_hlayers-1):
            layers.extend([nn.Linear(self.hlayer_sizes[i], self.hlayer_sizes[i+1]), self.act_fn])
        layers.append(nn.Linear(self.hlayer_sizes[-1], self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

    