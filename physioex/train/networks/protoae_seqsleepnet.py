from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SleepAEPrototypeModule
from physioex.train.networks.seqsleepnet import EpochEncoder, SequenceEncoder

module_config = dict()


class PrototypeAESeqSleepNet(SleepAEPrototypeModule):
    def __init__(self, module_config=module_config):
        super(PrototypeAESeqSleepNet, self).__init__(Net(module_config), module_config)


class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.L = module_config["seq_len"]
        self.nchan = module_config["in_channels"]
        self.T = module_config["T"]
        self.F = module_config["F"]
        self.latent_dim = module_config["latent_dim"]
        
        self.epoch_encoder = EpochEncoder(module_config)
        self.sequence_encoder = SequenceEncoder(module_config)

        self.lin_encode = nn.Linear(2 * module_config["seqnhidden2"], self.latent_dim)
        self.lin_decode = nn.Linear(self.latent_dim, 2 * module_config["seqnhidden2"])

        self.layer_norm = nn.LayerNorm(self.latent_dim)
        
        self.sequence_decoder = SequenceDecoder(module_config)
        self.epoch_decoder = EpochDecoder(module_config)
        self.classifier = Classifier(module_config)
        
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
        x = self.layer_norm(x)
        #print("after lin encode:", x.size())
        return x

    def decode(self, x):
        batch = x.size(0) #x.shape = [b, l, latent_dim]
        x = self.lin_decode(x)
        #print("after lin decode:", x.size())
        x = self.sequence_decoder(x)
        #print("after sequence decoder:", x.shape)
        x = x.reshape(batch * self.L, -1) #x.shape = [b*l, 128]
        #print("reshape:", x.size())
        x = self.epoch_decoder(x)
        #print("after epoch decoder:", x.size())
        x = x.reshape(batch, self.L, self.nchan, self.T, self.F)
        return x
    
    def classify(self, z):
        x = self.classifier(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        y = self.classify(z)
        x_hat = self.decode(z)
        return x_hat, z, y
    
class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.n_prototypes = config["n_prototypes"]
        self.n_classes = config["n_classes"]
        self.latent_dim = config["latent_dim"]
        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes, self.latent_dim))
        self.W = nn.Linear(self.n_prototypes, self.n_classes, bias=False)
        self.central_epoch = int((config["seq_len"] - 1) / 2)
        
    def forward(self, z):
        z = z[:, self.central_epoch, :]
        distances_p_to_z = torch.cdist(self.prototypes, z) #**2
        distances_p_to_z = distances_p_to_z.transpose(0, 1)
        y = self.W(distances_p_to_z)
        return y

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
        #[96, 128]
        #[96, 1, 29, 129]
        self.nchan = config["in_channels"]
        self.D = config["D"]
        self.T = config["T"]
        self.F = config["F"]
        self.input_size = config["seqnhidden1"]*2

        self.gru_epoch_decoder = nn.GRU(
            input_size=16,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.input_size, 16 * 8 * 16)
        
        # Strati convoluzionali trasposti
        self.conv_transpose1 = nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose2 = nn.ConvTranspose2d(8, 4, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose3 = nn.ConvTranspose2d(4, 1, kernel_size=(3, 5), stride=(2, 2), padding=(1, 0), output_padding=(0,0))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        #print(x.size())
        x = self.leaky_relu(self.fc1(x))
        #print(x.size())
        x = x.view(-1, 8, 16)
        #print(x.size())
        x, _ = self.gru_epoch_decoder(x) # [batch*L, T, F*nchan]
        #print(x.size())
        forward_output = x[:, :, :16]
        backward_output = x[:, :, 16:]
        x = forward_output + backward_output

        #print(x.size())
        # Reshape per adattarsi al primo strato conv2d trasposto
        x = x.view(-1, 16, 8, 16)
        #print(x.size())
        # Strati convoluzionali trasposti
        x = self.leaky_relu(self.conv_transpose1(x))
        #print("t1", x.size())
        x = self.leaky_relu(self.conv_transpose2(x))
        #print("t2", x.size())
        x = self.conv_transpose3(x)
        #print("t3", x.size())
        
        return x


    