from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SelfSupervisedSleepModule
from physioex.train.networks.utils.loss import ReconstructionLoss

module_config = dict()

class AutoEncoderFullyConnected(SelfSupervisedSleepModule):
    def __init__(self, module_config=module_config):
        module_config.update(
            {
                "latent_dim": 32,
                "epoch_encode_dim": 128, #latent dim before sequence encoding
                "T": 29,
                "F": 129,
                "alpha1": 1.0,
                "alpha2": 0.1,
                "alpha3": 0.3,
                "alpha4": 0.1
            }
        )

        super(AutoEncoderFullyConnected, self).__init__(Net(module_config), module_config)

        self.loss = ReconstructionLoss(
            alpha1=module_config["alpha1"],
            alpha2=module_config["alpha2"],
            alpha3=module_config["alpha3"],
            alpha4=module_config["alpha4"]
        )
        self.factor_names = ["loss", "mse", "std-pen", "std-pen-T", "std-pen-F"]
        self.metrics = None
        self.metric_names = None

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.encoder = Encoder(module_config)
        self.decoder = Decoder(module_config)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y

    def encode(self, x):
        return self.encoder(x)

class Encoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.L = config["sequence_length"]
        self.nchan = config["in_channels"]
        self.T = config["T"]
        self.F = config["F"]
        self.input_dim = self.nchan * self.T * self.F
        self.epoch_encode_dim = config["epoch_encode_dim"]
        self.latent_dim = config["latent_dim"]

        self.epoch_encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, self.epoch_encode_dim), nn.LeakyReLU()
        )
        
        self.sequence_encoder = nn.Sequential(
            nn.Linear(self.epoch_encode_dim * self.L, 128), nn.LeakyReLU(),
            nn.Linear(128, self.latent_dim * self.L), nn.LeakyReLU()
        )                                                           

        self.lin_encode = nn.Linear(self.latent_dim, self.latent_dim)
        self.layer_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, x): # x: [batch, L, nchan, T, F]
        x = x.view(-1, self.nchan * self.T * self.F) # [batch, input_dim]
        x = self.epoch_encoder(x) # [batch * L, epoch_encode_dim]
        x = x.view(-1, self.L * self.epoch_encode_dim) # [batch, L * epoch_encode_dim]
        x = self.sequence_encoder(x) # [batch, L * latent_dim]
        x = x.view(-1, self.L, self.latent_dim) # [batch, L, latent_dim]
        x = self.lin_encode(x) # [batch, L, latent_dim]
        x = self.layer_norm(x) # [batch, L, latent_dim]
        return x

class Decoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.L = config["sequence_length"]
        self.nchan = config["in_channels"]
        self.T = config["T"]
        self.F = config["F"]
        self.latent_dim = config["latent_dim"]
        self.epoch_encode_dim = config["epoch_encode_dim"]
        self.output_dim = self.nchan * self.T * self.F

        self.lin_decode = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.LeakyReLU())

        self.sequence_decoder = nn.Sequential(
            nn.Linear(self.latent_dim * self.L, 128), nn.LeakyReLU(),
            nn.Linear(128, self.epoch_encode_dim * self.L), nn.LeakyReLU()
        )
        self.epoch_decoder = nn.Sequential(
            nn.Linear(self.epoch_encode_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, x): # x: [batch, L, latent_dim]
        x = self.lin_decode(x) # [batch, L, latent_dim]
        x = x.view(-1, self.L * self.latent_dim) # [batch, L * latent_dim]
        x = self.sequence_decoder(x) # [batch, L * epoch_encode_dim]
        x = x.view(-1, self.epoch_encode_dim) # [batch * L, epoch_encode_dim]
        x = self.epoch_decoder(x) # [batch * L, input_dim]
        x = x.view(-1, self.L, self.nchan, self.T, self.F) # [batch, L, nchan, T, F]
        return x

    