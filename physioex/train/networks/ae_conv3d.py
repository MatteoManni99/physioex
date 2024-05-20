from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SleepAutoEncoderModule

module_config = dict()


class AutoEncoderConv3D(SleepAutoEncoderModule):
    def __init__(self, module_config=module_config):
        super(AutoEncoderConv3D, self).__init__(Net(module_config), module_config)

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
        self.epoch_encode_dim = config["epoch_encode_dim"]
        self.output_dim = config["latent_dim"]

        self.epoch_encoder = nn.Sequential(
            nn.Conv2d(self.nchan, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # Output: 32 x 15 x 65
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),         # Output: 64 x 8 x 33
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),          # Output: 32 x 8 x 33
            nn.Tanh()
        )

        self.sequence_encoder = nn.Sequential(
            nn.Conv3d(32, self.output_dim, kernel_size=(3, 5, 17), padding=(1, 0, 0)),
            nn.Tanh(),
            nn.Conv3d(32, self.output_dim, kernel_size=(3, 3, 11), padding=(1, 0, 0)),
            nn.Tanh(),
            nn.Conv3d(32, self.output_dim, kernel_size=(3, 2, 7), padding=(1, 0, 0)),
            nn.Tanh(),
        )

        self.lin_encode = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x):
        #print("Encoder input shape: ", x.shape)
        batch_size, L, channels, T, F = x.shape
        x = x.view(-1, channels, T, F)  # [batch*L, channels, T, F]
        #print("Encoder Reshaped for epoch encoding: ", x.shape)
        x = self.epoch_encoder(x)  # [batch*L, 32, 8, 32]
        #print("Encoder Epoch encoded shape: ", x.shape)
        _, encoded_channels, encoded_T, encoded_F = x.shape
        x = x.view(batch_size, L, encoded_channels, encoded_T, encoded_F)  # [batch, L, encoded_channels, T', F']
        #print("Encoder reshaped for sequence encoding: ", x.shape)
        x = x.permute(0, 2, 1, 3, 4)  # [batch, encoded_channels, L, T', F']
        #print("Encoder permute input shape: ", x.shape)
        x = self.sequence_encoder(x)
        #print("Encoder Sequence encoded shape: ", x.shape)
        x = x.permute(0, 2, 1, 3, 4)  # [batch, L, encoded_channels, T', F']
        #print("Encoder Permuted for linear encoding: ", x.shape)
        x = x.reshape(batch_size, L,  self.output_dim)
        #print("Encoder Reshaped for linear encoding: ", x.shape)
        x = self.lin_encode(x)
        #print("Encoder Linear encoded shape: ", x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.L = config["seq_len"]
        self.nchan = config["in_channels"]
        self.T = config["T"]
        self.F = config["F"]
        self.input_dim = config["latent_dim"]
        self.output_dim = self.nchan * self.T * self.F

        self.epoch_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),         
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),          
            nn.Tanh(),
            nn.ConvTranspose2d(32, self.nchan, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        self.sequence_decoder = nn.Sequential(
            nn.ConvTranspose3d(self.input_dim, 32, kernel_size=(3, 2, 7), padding=(1, 0, 0)),
            nn.Tanh(),
            nn.ConvTranspose3d(self.input_dim, 32, kernel_size=(3, 3, 11), padding=(1, 0, 0)),
            nn.Tanh(),
            nn.ConvTranspose3d(self.input_dim, 32, kernel_size=(3, 5, 17), padding=(1, 0, 0)),
            nn.Tanh(),
        )

        self.lin_decode = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, x):
        #print("Decoder input shape: ", x.shape)
        batch_size, L, _ = x.shape
        x = self.lin_decode(x)
        #print("Decoder Linear decoded shape: ", x.shape)
        x = x.view(batch_size, L, self.input_dim, 1, 1)
        #print("Decoder Reshaped for sequence decoding: ", x.shape)
        x = x.permute(0, 2, 1, 3, 4)  # [batch, input_dim, L, 1, 1]
        #print("Decoder permute for sequence decoding: ", x.shape)
        x = self.sequence_decoder(x)
        #print("Decoder Sequence decoded shape: ", x.shape)
        _, decoded_channels, _, decoded_T, decoded_F = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        #print("Decoder permute for epoch decoding: ", x.shape)
        x = x.reshape(-1, decoded_channels, decoded_T, decoded_F)  # [batch*L, 32, T', F']
        #print("Decoder Reshaped for epoch decoding: ", x.shape)
        x = self.epoch_decoder(x)
        #print("Decoder Epoch decoded shape: ", x.shape)
        x = x.view(batch_size, L, self.nchan, self.T, self.F)  # [batch, L, channels, T, F]
        #print("Decoder reshaped for permute: ", x.shape)
        return x