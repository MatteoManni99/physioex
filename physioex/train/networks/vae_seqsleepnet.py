from math import floor
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SelfSupervisedSleepModule
from physioex.train.networks.seqsleepnet import EpochEncoder, SequenceEncoder
from physioex.train.networks.utils.loss import ReconstructionLoss
module_config = dict()


class VAESeqSleepNet(SelfSupervisedSleepModule):
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
                "beta": 0.000002
            }
        )
        super(VAESeqSleepNet, self).__init__(Net(module_config), module_config)
        # self.L = module_config["seq_len"]
        # self.central = floor(self.L/2)
        # self.penalty = True
        self.beta = module_config["beta"]
        self.loss = ReconstructionLoss(
            alpha1=module_config["alpha1"],
            alpha2=module_config["alpha2"],
            alpha3=module_config["alpha3"],
            alpha4=module_config["alpha4"],
        )
        self.factor_names = ["loss", "mse", "std-pen", "std-pen-T", "std-pen-F", "kl_div"]

    #its necessary to redefine the computer_loss method because the basic one does not support the return of mean and logvar
    def compute_loss(
        self,
        embeddings,
        inputs,
        outputs,
        log: str = "train",
        log_metrics: bool = False,
    ):  
        input_hat, mean, logvar = outputs
        
        loss_list = self.loss(None, inputs, input_hat)

        kl_div = self.klDivergence(mean, logvar)
        
        #passages to avoid to interrupt the computational graph
        loss = loss_list[0] + self.beta * kl_div
        loss_list = list(loss_list) + [kl_div]
        loss_list[0] = loss.item()

        #log all loss factors in the loss_list
        for i, factor in enumerate(loss_list):
            self.log(f"{log}_{self.factor_names[i]}", factor, prog_bar=True)

        return loss

    def klDivergence(self, mean, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())  # Loss KL Divergence
        return kld_loss
    

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.L = module_config["sequence_length"]
        self.nchan = module_config["in_channels"]
        self.T = module_config["T"]
        self.F = module_config["F"]
        self.central = floor(self.L/2)
        
        self.epoch_encoder = EpochEncoder(module_config)
        self.sequence_encoder = SequenceEncoder(module_config)

        self.lin_mean = nn.Linear(2 * module_config["seqnhidden2"], module_config["latent_dim"])
        self.lin_logvar = nn.Linear(2 * module_config["seqnhidden2"], module_config["latent_dim"])

        self.lin_decode = nn.Linear(module_config["latent_dim"], 2 * module_config["seqnhidden2"])
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
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, x):
        batch = x.size(0) #x.shape = [b, l, latent_dim]
        x = self.leaky_relu(self.lin_decode(x)) #x.shape = [b, l, 128] l=1
        x = self.sequence_decoder(x)
        x = self.epoch_decoder(x)
        x = x.reshape(batch, self.L, self.nchan, self.T, self.F)
        return x

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decode(z)
        return z, (x_reconstructed, mean, logvar)
    

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

        self.conv_transpose1 = nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose2 = nn.ConvTranspose2d(8, 4, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose3 = nn.ConvTranspose2d(4, 1, kernel_size=(3, 5), stride=(2, 2), padding=(1, 0), output_padding=(0,0))

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = x.view(-1, 8, 16)
        x, _ = self.gru_epoch_decoder(x) # [batch*L, T, F*nchan]
        forward_output = x[:, :, :16]
        backward_output = x[:, :, 16:]
        x = forward_output + backward_output

        x = x.view(-1, 16, 8, 16)
        # Strati convoluzionali trasposti
        x = self.leaky_relu(self.conv_transpose1(x))
        x = self.leaky_relu(self.conv_transpose2(x))
        x = self.conv_transpose3(x)
        
        return x
    