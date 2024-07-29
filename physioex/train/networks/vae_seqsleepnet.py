from math import floor
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SleepAutoEncoderModule
from physioex.train.networks.seqsleepnet import EpochEncoder, SequenceEncoder

module_config = dict()


class VAESeqSleepNet(SleepAutoEncoderModule):
    def __init__(self, module_config=module_config):
        super(VAESeqSleepNet, self).__init__(Net(module_config), module_config)
        self.L = module_config["seq_len"]
        self.central = floor(self.L/2)
        self.penalty = True 
    #its necessary to redefine the computer_loss method because the basic one does not support the return of mean and logvar
    def compute_loss(
        self,
        inputs,
        outputs,
        log: str = "train",
    ):  
        #selecting only the central input
        #inputs = inputs[:, self.central, :].unsqueeze(1) #<-- decode di una sola epoca
        input_hat, mean, logvar = outputs
        mse = self.loss(inputs, input_hat)
        std_penalty, std_penalty_T, std_penalty_F = self.std_penalty(inputs, input_hat)
        

        #check if penalty is active and add it to the loss
        if(self.penalty):
            loss_reconstruction = 2 * mse + 0.5 * std_penalty + 0.8* std_penalty_T + 0.2 * std_penalty_F
        else:
            loss_reconstruction = 2 * mse
        
        kl_div = self.klDivergence(mean, logvar)
        
        loss = loss_reconstruction + 0.000002 * self.klDivergence(mean, logvar)
        #loss = loss_reconstruction

        self.log(f"{log}_loss_tot", loss, prog_bar=True)
        self.log(f"{log}_loss", mse, prog_bar=True)
        self.log(f"{log}_kl_div", kl_div, prog_bar=True)
        self.log(f"{log}_std_penalty", std_penalty, prog_bar=True)
        self.log(f"{log}_std_penalty_T", std_penalty_T, prog_bar=True)
        self.log(f"{log}_std_penalty_F", std_penalty_F, prog_bar=True)
        return loss

    def klDivergence(self, mean, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())  # Loss KL Divergence
        return kld_loss
    

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.L = module_config["seq_len"]
        self.nchan = module_config["in_channels"]
        self.T = module_config["T"]
        self.F = module_config["F"]
        self.central = floor(self.L/2)
        
        self.epoch_encoder = EpochEncoder(module_config)
        self.sequence_encoder = SequenceEncoder(module_config)

        self.lin_mean = nn.Linear(2 * module_config["seqnhidden2"], module_config["latent_dim"])
        self.lin_logvar = nn.Linear(2 * module_config["seqnhidden2"], module_config["latent_dim"])

        self.lin_decode = nn.Linear(module_config["latent_dim"], 2 * module_config["seqnhidden2"])
        # self.lin_decode2 = nn.Linear(2 * module_config["seqnhidden2"], 128)
        # self.lin_decode3 = nn.Linear(128, 128)
        #self.layer_norm = nn.LayerNorm(module_config["latent_dim"])
        self.leaky_relu = nn.LeakyReLU()
        self.sequence_decoder = SequenceDecoder(module_config)
        self.epoch_decoder = EpochDecoder(module_config)

    def encode(self, x):
        batch = x.size(0)
        x = x.reshape(-1, self.nchan, self.T, self.F)
        x = self.epoch_encoder(x)
        x = x.reshape(batch, self.L, -1)
        x = self.sequence_encoder.encode(x) #x.shape = [b, l, seqnhidden2 * 2]
        mean = self.lin_mean(x)
        logvar = self.lin_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        #print("mean:", mean[0][self.central], "std:", std[0][self.central])
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, x):
        batch = x.size(0) #x.shape = [b, l, latent_dim]
        x = self.leaky_relu(self.lin_decode(x)) #x.shape = [b, l, 128] l=1
        #print("after lin decode:", x.size())
        x = self.sequence_decoder(x)
        # print("after sequence decoder:", x.shape)
        #---AL POSTO DEL SEQUENCE DECODER ORIGINARIO--- scrivere su latex prima di cancellare (non funziona)
        # x = self.leaky_relu(self.lin_decode2(x)) # x.shape = [b, l, 128] l=1
        # #print("after lin decode2:", x.size())
        # x = self.leaky_relu(self.lin_decode3(x)) # x.shape = [b, l, 128]  l=1
        #print("after lin decode3:", x.size())
        #x = x.reshape(batch * self.L, -1) #x.shape = [b*l, 128]
        #print("reshape:", x.size())
        x = self.epoch_decoder(x)
        #print("after epoch decoder:", x.size())
        x = x.reshape(batch, self.L, self.nchan, self.T, self.F)
        #x = x.reshape(batch, 1, self.nchan, self.T, self.F) #<-- decode di una sola epoca
        return x

    def forward(self, x):
        mean, logvar = self.encode(x)
        #selecting only the central encoding
        # mean = mean[:, self.central, :]
        # logvar = logvar[:, self.central, :]
        # mean = mean.unsqueeze(1)
        # logvar = logvar.unsqueeze(1)
        z = self.reparameterize(mean, logvar)
        #self.count_zeros(z)
        x_reconstructed = self.decode(z)
        return (x_reconstructed, mean, logvar)

    def count_zeros(self, x):
        num = (torch.abs(x) < 0.0001).sum(dim=(1, 2))
        num = num.float().mean()
        print("n.zeros: ", num)
    

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
        self.gru_epoch_decoder = nn.GRU(
            input_size=16,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(128, 16 * 8 * 16)
        #self.fc2 = nn.Linear(1*29*64, 1*29*129)
        #self.fc3 = nn.Linear(256, 16 * 4 * 16)  # Adattare per il successivo reshaping
        
        # Strati convoluzionali trasposti
        self.conv_transpose1 = nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0,0))
        #self.conv_transpose12 = nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose2 = nn.ConvTranspose2d(8, 4, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose3 = nn.ConvTranspose2d(4, 1, kernel_size=(3, 5), stride=(2, 2), padding=(1, 0), output_padding=(0,0))

    def forward(self, x):
        # Livelli fully connected
        x = self.leaky_relu(self.fc1(x))
        x = x.view(-1, 8, 16)
        x, _ = self.gru_epoch_decoder(x) # [batch*L, T, F*nchan]
        forward_output = x[:, :, :16]
        backward_output = x[:, :, 16:]
        x = forward_output + backward_output

        # Reshape per adattarsi al primo strato conv2d trasposto
        x = x.view(-1, 16, 8, 16)
        # Strati convoluzionali trasposti
        x = self.leaky_relu(self.conv_transpose1(x))
        x = self.leaky_relu(self.conv_transpose2(x))
        x = self.conv_transpose3(x)
        
        return x
    