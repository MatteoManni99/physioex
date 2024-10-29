from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.selfsupervised import SelfSupervisedSleepModule
from physioex.train.networks.seqsleepnet import EpochEncoder, SequenceEncoder
from physioex.train.networks.utils.loss import ReconstructionLoss

module_config = dict()


class AutoEncoderSeqSleepNet(SelfSupervisedSleepModule):
    def __init__(self, module_config=module_config):

        module_config.update(
            {
                "T": 29,
                "F": 129,
                "D": 32,
                "nfft": 256,
                "lowfreq": 0,
                "highfreq": 50,
                "seqnhidden1": 64,
                "seqnlayer1": 1,
                "attentionsize1": 32,
                "seqnhidden2": 64,
                "seqnlayer2": 1,
                "latent_dim": 32,
                "alpha1": 1.0,
                "alpha2": 0.1,
                "alpha3": 0.3,
                "alpha4": 0.1,
            }
        )
             
        super(AutoEncoderSeqSleepNet, self).__init__(Net(module_config), module_config)

        self.loss = ReconstructionLoss(
            alpha1=module_config["alpha1"],
            alpha2=module_config["alpha2"],
            alpha3=module_config["alpha3"],
            alpha4=module_config["alpha4"]
        )
        self.loss_factor_names = ["loss", "mse", "std-pen", "std-pen-T", "std-pen-F"]
        self.metrics = None
        self.metric_names = None

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.L = module_config["sequence_length"]
        self.nchan = module_config["in_channels"]
        self.T = module_config["T"]
        self.F = module_config["F"]

        self.epoch_encoder = EpochEncoder(module_config)
        self.sequence_encoder = SequenceEncoder(module_config)

        self.lin_encode = nn.Linear(2 * module_config["seqnhidden2"], module_config["latent_dim"])
        self.lin_decode = nn.Linear(module_config["latent_dim"], 2 * module_config["seqnhidden2"])

        self.layer_norm = nn.LayerNorm(module_config["latent_dim"])
        
        self.sequence_decoder = SequenceDecoder(module_config)
        self.epoch_decoder = EpochDecoder(module_config)

    def encode(self, x):
        batch = x.size(0)
        x = x.view(-1, self.nchan, self.T, self.F)
        x = self.epoch_encoder(x)
        x = x.view(batch, self.L, -1)
        x = self.sequence_encoder.encode(x)
        x = self.lin_encode(x)
        x = self.layer_norm(x)
        return x

    def decode(self, x):
        batch = x.size(0)
        x = self.lin_decode(x)
        x = self.sequence_decoder(x)
        x = x.reshape(batch * self.L, -1) #x.shape = [b*l, 128]
        x = self.epoch_decoder(x)
        x = x.view(batch, self.L, self.nchan, self.T, self.F)
        return x

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return z, y

class SequenceDecoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.gru_seq_decoder = nn.LSTM(
            input_size=config["seqnhidden2"] * 2,
            hidden_size=config["seqnhidden1"],
            num_layers=config["seqnlayer2"],
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        x, _= self.gru_seq_decoder(x)
        return x


class EpochDecoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.nchan = config["in_channels"]
        self.D = config["D"]
        self.T = config["T"]
        self.F = config["F"]
        self.gru_epoch_decoder = nn.GRU(
            input_size=16,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(128, 16 * 8 * 16)
        
        #filter bank specular part
        self.conv_transpose1 = nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose2 = nn.ConvTranspose2d(8, 4, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose3 = nn.ConvTranspose2d(4, 1, kernel_size=(3, 5), stride=(2, 2), padding=(1, 0), output_padding=(0,0))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = x.view(-1, 8, 16)
        x, _ = self.gru_epoch_decoder(x)
        forward_output = x[:, :, :16]
        backward_output = x[:, :, 16:]
        x = forward_output + backward_output
        
        x = x.view(-1, 16, 8, 16)
        x = self.leaky_relu(self.conv_transpose1(x))
        x = self.leaky_relu(self.conv_transpose2(x))
        x = self.conv_transpose3(x)
        
        return x