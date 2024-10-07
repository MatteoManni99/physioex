from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SelfSupervisedSleepModule
from physioex.train.networks.seqsleepnet import EpochEncoder, SequenceEncoder
from physioex.train.networks.utils.loss import SemiSupervisedLoss
import torchmetrics as tm

module_config = dict()


class PrototypeAESeqSleepNet(SelfSupervisedSleepModule):
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
                "lambda1": 6,
                "lambda2": 6,
                "lambda3": 1,
                "lambda4": 2,

                "n_prototypes": 15,
            }
        )

        super(PrototypeAESeqSleepNet, self).__init__(Net(module_config), module_config)
        self.central_epoch = int((module_config["sequence_length"] - 1) / 2)

        self.loss = SemiSupervisedLoss(
            alpha1=module_config["alpha1"],
            alpha2=module_config["alpha2"],
            alpha3=module_config["alpha3"],
            alpha4=module_config["alpha4"],
            lambda1=module_config["lambda1"],
            lambda2=module_config["lambda2"],
            lambda3=module_config["lambda3"],
            lambda4=module_config["lambda4"],
        )
        print(self.device)
        self.factor_names = ["loss", "cel", "r1", "r2", "rec_loss", "mse", "std_pen", "std_pen_T", "std_pen_F"]
        self.f1 = tm.F1Score(task="multiclass", num_classes=module_config["n_classes"], average="weighted")
        self.Mf1 = tm.F1Score(task="multiclass", num_classes=module_config["n_classes"], average="macro")
        self.metrics = [self.f1, self.Mf1]
        self.metric_names = ["f1", "Mf1"]
    
    def compute_loss(
        self,
        inputs,
        inputs_hat,
        embeddings,
        labels,
        pred,
        log: str = "train",
        log_metrics: bool = False,
    ):
        inputs = inputs[:, self.central_epoch, 0, :, :]
        inputs_hat = inputs_hat[:, self.central_epoch, 0, :, :]
        embeddings = embeddings[:, self.central_epoch, :]
        labels = labels[:, self.central_epoch]
        
        #loss_list = [loss, loss1, loss2, ...] where loss is the total loss
        loss_list = self.loss(
            embeddings,
            self.nn.classifier.prototypes,
            pred,
            labels, 
            inputs,
            inputs_hat
        )

        #log all loss factors in the loss_list
        for i, factor in enumerate(loss_list):
            self.log(f"{log}_{self.factor_names[i]}", factor, prog_bar=True)
        
        #log the metrics if log_metrics is True and the metrics are defined 
        if log_metrics and self.metrics is not None:
            for i, metric in enumerate(self.metrics):
                self.log(f"{log}_{self.metric_names[i]}", metric(pred, labels), prog_bar=True)
        
        return loss_list[0]
            
    def training_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, embeddings, pred = self.forward(x)
        return self.compute_loss(x, x_hat, embeddings, labels, pred)

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, embeddings, pred = self.forward(x)
        return self.compute_loss(x, x_hat, embeddings, labels, pred, log="val", log_metrics=True)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, embeddings, pred = self.forward(x)
        return self.compute_loss(x, x_hat, embeddings, labels, pred, log="test", log_metrics=True)

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.L = module_config["sequence_length"]
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
        x = x.view(-1, self.nchan, self.T, self.F)
        x = self.epoch_encoder(x)
        x = x.view(batch, self.L, -1)
        x = self.sequence_encoder.encode(x)
        x = self.lin_encode(x)
        x = self.layer_norm(x)
        return x

    def decode(self, x):
        batch = x.size(0) #x.shape = [b, l, latent_dim]
        x = self.lin_decode(x)
        x = self.sequence_decoder(x)
        x = x.reshape(batch * self.L, -1) #x.shape = [b*l, 128]
        x = self.epoch_decoder(x)
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
        self.central_epoch = int((config["sequence_length"] - 1) / 2)
        
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
        x = self.leaky_relu(self.fc1(x))
        x = x.view(-1, 8, 16)
        x, _ = self.gru_epoch_decoder(x) # [batch*L, T, F*nchan]
        forward_output = x[:, :, :16]
        backward_output = x[:, :, 16:]
        x = forward_output + backward_output

        x = x.view(-1, 16, 8, 16)
        x = self.leaky_relu(self.conv_transpose1(x))
        x = self.leaky_relu(self.conv_transpose2(x))
        x = self.conv_transpose3(x)
        
        return x


    