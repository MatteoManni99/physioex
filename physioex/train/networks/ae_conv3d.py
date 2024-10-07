from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SelfSupervisedSleepModule
from physioex.train.networks.utils.loss import ReconstructionLoss

module_config = dict()


class AutoEncoderConv3D(SelfSupervisedSleepModule):
    def __init__(self, module_config=module_config):
        module_config.update(
            {
                "latent_dim": 32,
                "T": 29,
                "F": 129,
                "alpha1": 1.0,
                "alpha2": 0.1,
                "alpha3": 0.3,
                "alpha4": 0.1
            }
        )

        super(AutoEncoderConv3D, self).__init__(Net(module_config), module_config)

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
        self.latent_dim = config["latent_dim"]

        #expected input shape: [batch * L, channels, T, F] = [batch * L, channels, 29, 129]
        self.epoch_encoder = nn.Sequential(
            nn.Conv2d(self.nchan, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), # Output: [batch * L, 32, 15, 65]
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),         # Output: [batch * L, 64, 8, 33]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),         # Output: [batch * L, 32, 8, 33]
            nn.LeakyReLU()
        )

        # expected input shape: [batch, 32, l, 8, 33]
        self.sequence_encoder = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 5, 17), padding=(1, 0, 0)), # Output: [batch, 32, l, 4, 17]
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 11), padding=(1, 0, 0)), # Output: [batch, 32, l, 2, 7]
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 2, 7), padding=(1, 0, 0)), # Output: [batch, 32, l, 1, 1]
            nn.LeakyReLU(),
        )

        self.lin_encode = nn.Linear(32, self.latent_dim)
        self.layer_norm_encode = nn.LayerNorm(self.latent_dim)

    def forward(self, x):
        batch_size, L, channels, T, F = x.shape
        x = x.view(-1, channels, T, F)  # [batch*L, channels, T, F]
        x = self.epoch_encoder(x)  # [batch*L, 32, 8, 32]
        _, encoded_channels, encoded_T, encoded_F = x.shape
        x = x.view(batch_size, L, encoded_channels, encoded_T, encoded_F) # [batch, L, 32, 8, 32]
        x = x.permute(0, 2, 1, 3, 4) # [batch, 32, L, 8, 32]
        x = self.sequence_encoder(x) # [batch, 32, L, 1, 1]
        x = x.permute(0, 2, 1, 3, 4) # [batch, L, 32, 1, 1]
        x = x.view(batch_size, L,  self.latent_dim) # [batch, L, 32]
        x = self.lin_encode(x)
        x = self.layer_norm_encode(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.L = config["sequence_length"]
        self.nchan = config["in_channels"]
        self.T = config["T"]
        self.F = config["F"]
        self.latent_dim = config["latent_dim"]

        self.lin_decode = nn.Sequential(nn.Linear(self.latent_dim, 32), nn.LeakyReLU())

        # expected input shape: [batch, 32, L, 1, 1]
        self.sequence_decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=(3, 2, 7), padding=(1, 0, 0)), #Output: [batch, 32, L, 2, 7]
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32, 32, kernel_size=(3, 3, 11), padding=(1, 0, 0)), #Output: [batch, 32, L, 4, 17]
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32, 32, kernel_size=(3, 5, 17), padding=(1, 0, 0)), #Output: [batch, 32, L, 8, 33]
            nn.LeakyReLU(),
        )

        # expected input shape: [batch * L, 32, 8, 33]
        self.epoch_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # Output: [batch * L, 64, 8, 33]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), # Output: [batch * L, 32, 8, 33]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, self.nchan, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), # Output: [batch * L, nchan, 29, 129] = [batch * L, nchan, T, F]
        )

    def forward(self, x):
        batch_size, L, _ = x.shape
        x = self.lin_decode(x) # [batch, L, 32]
        x = x.view(batch_size, L, self.latent_dim, 1, 1) # [batch, L, 32, 1, 1]
        x = x.permute(0, 2, 1, 3, 4)  # [batch, 32, L, 1, 1]
        x = self.sequence_decoder(x) # [batch, 32, L, 8, 33]
        _, decoded_channels, _, decoded_T, decoded_F = x.shape # [batch, 32, L, 8, 33]
        x = x.permute(0, 2, 1, 3, 4) # [batch, L, 32, 8, 33]
        x = x.view(-1, decoded_channels, decoded_T, decoded_F)  # [batch*L, 32, 8, 33]
        x = self.epoch_decoder(x) # [batch*L, channels, T, F]
        x = x.view(batch_size, L, self.nchan, self.T, self.F)  # [batch, L, channels, T, F]
        return x