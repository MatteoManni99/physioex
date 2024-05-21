from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SleepAutoEncoderModule
from physioex.train.networks.seqsleepnet import EpochEncoder, SequenceEncoder

module_config = dict()


class AutoEncoderSeqSleepNet(SleepAutoEncoderModule):
    def __init__(self, module_config=module_config):
        super(AutoEncoderSeqSleepNet, self).__init__(Net(module_config), module_config)

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.L = module_config["seq_len"]
        self.nchan = module_config["in_channels"]
        self.T = module_config["T"]
        self.F = module_config["F"]

        self.epoch_encoder = EpochEncoder(module_config)
        self.sequence_encoder = SequenceEncoder(module_config)

        self.lin_encode = nn.Linear(2 * module_config["seqnhidden2"], module_config["latent_dim"])
        self.lin_decode = nn.Linear(module_config["latent_dim"], 2 * module_config["seqnhidden2"])

        self.sequence_decoder = SequenceDecoder(module_config)
        self.epoch_decoder = EpochDecoder(module_config)

    def encode(self, x):
        batch = x.size(0)
        #print("input:", x.size())
        x = x.view(-1, self.nchan, self.T, self.F)
        #print("reshape:", x.size())
        x = self.epoch_encoder(x)
        #print("after epoch encoder:", x.size())
        x = x.view(batch, self.L, -1)
        #print("reshape:", x.size())
        x = self.sequence_encoder.encode(x)
        #print("after sequence encoder:", x.size())
        x = self.lin_encode(x)
        #print("after lin encode:", x.size())
        return x

    def decode(self, x):
        batch = x.size(0)
        x = self.lin_decode(x)
        #print("after lin decode:", x.size())
        x = self.sequence_decoder(x)
        #print("after sequence decoder:", x.shape)
        x = x.reshape(batch * self.L, -1)
        #print("reshape:", x.size())
        x = self.epoch_decoder(x)
        #print("after epoch decoder:", x.size())
        x = x.view(batch, self.L, self.nchan, self.T, self.F)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class SequenceDecoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.LSTM = nn.GRU(
            input_size=config["seqnhidden2"] * 2,
            hidden_size=config["seqnhidden1"],
            num_layers=config["seqnlayer2"],
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        x, _= self.LSTM(x)
        return x

class EpochDecoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        #[96, 128]
        #[96, 1, 29, 129]
        # self.F2_birnn = nn.LSTM(
        #     input_size=config["seqnhidden1"],
        #     hidden_size=config["D"] * config["in_channels"],
        #     num_layers=config["seqnlayer1"],
        #     batch_first=True,
        #     bidirectional=True,
        # )
        self.transposeConvNet = TransposeConvNet(input_dim=config["seqnhidden1"]*2, nchan=config["in_channels"], T=config["T"], F=config["F"])

    def forward(self, x):  
        x = self.transposeConvNet(x)
        return x

class TransposeConvNet(nn.Module):
    def __init__(self, input_dim, nchan, T, F):
        super(TransposeConvNet, self).__init__()
        self.input_dim = input_dim
        self.nchan = nchan
        self.T = T
        self.F = F

        # First ConvTranspose2d layer parameters
        self.conv_transpose1 = nn.ConvTranspose2d(
            in_channels=self.input_dim, 
            out_channels=int(self.input_dim/2), 
            kernel_size=(5, 5),
            stride=(2, 2)
        )
        
        # Calculate intermediate dimensions after first layer
        self.intermediate_T1 = (1 - 1) * 2 + 5
        self.intermediate_F1 = (1 - 1) * 2 + 5

        # Second ConvTranspose2d layer parameters
        self.conv_transpose2 = nn.ConvTranspose2d(
            in_channels=int(self.input_dim/2), 
            out_channels=int(self.input_dim/4), 
            kernel_size=(5, 5), 
            stride=(2, 2)
        )
        
        # Calculate intermediate dimensions after second layer
        self.intermediate_T2 = (self.intermediate_T1 - 1) * 2 + 5
        self.intermediate_F2 = (self.intermediate_F1 - 1) * 2 + 5

        # Third ConvTranspose2d layer parameters
        self.conv_transpose3 = nn.ConvTranspose2d(
            in_channels=int(self.input_dim/4),
            out_channels=self.nchan, 
            kernel_size=(self.T - self.intermediate_T2 + 1, self.F - self.intermediate_F2 + 1), 
            stride=(1, 1)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim, 1, 1) # Reshape to match the input of ConvTranspose2d
        x = self.conv_transpose1(x)
        x = nn.Tanh()(x)
        x = self.conv_transpose2(x)
        x = nn.Tanh()(x)
        x = self.conv_transpose3(x)
        return x
    