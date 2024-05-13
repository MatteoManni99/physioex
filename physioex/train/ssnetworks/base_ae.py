from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

activation_function = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "leakyrelu": nn.LeakyReLU(), "elu": nn.ELU(), "selu": nn.SELU(), "gelu": nn.GELU(), "none": nn.Identity()}

class Encoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.act_fn = activation_function[config["encoder_act_fun"]]
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
        self.act_fn = activation_function[config["decoder_act_fun"]]
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

class BaseAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        config: Dict,
    ):
        super().__init__()
        self.loss_function = nn.MSELoss(reduction="mean")
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.save_hyperparameters()

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = self.loss_function(x, x_hat)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
