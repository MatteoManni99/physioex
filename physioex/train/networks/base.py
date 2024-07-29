from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
import torch.nn.functional as Fun
from physioex.train.networks.utils.loss import Reconstruction

class SleepModule(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SleepModule, self).__init__()
        self.save_hyperparameters()
        self.nn = nn

        self.n_classes = config["n_classes"]

        if self.n_classes > 1:
            # classification experiment
            self.acc = tm.Accuracy(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
            self.f1 = tm.F1Score(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
            self.ck = tm.CohenKappa(task="multiclass", num_classes=config["n_classes"])
            self.pr = tm.Precision(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
            self.rc = tm.Recall(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
        elif self.n_classes == 1:
            # regression experiment
            self.mse = tm.MeanSquaredError()
            self.mae = tm.MeanAbsoluteError()
            self.r2 = tm.R2Score()

        # loss
        self.loss = config["loss_call"](config["loss_params"])
        self.module_config = config

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=1e-4,
            weight_decay=1e-3,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="max",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )

        return {
            "optimizer": self.opt,
            "lr_scheduler": {"scheduler": self.scheduler, "monitor": "val_acc"},
        }

    def forward(self, x):
        return self.nn(x)

    def encode(self, x):
        return self.nn.encode(x)

    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        # print(targets.size())
        batch_size, seq_len, n_class = outputs.size()

        embeddings = embeddings.reshape(batch_size * seq_len, -1)
        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)

        if self.n_classes > 1:
            loss = self.loss(embeddings, outputs, targets)

            self.log(f"{log}_loss", loss, prog_bar=True)
            self.log(f"{log}_acc", self.acc(outputs, targets), prog_bar=True)
            self.log(f"{log}_f1", self.f1(outputs, targets), prog_bar=True)
        else:
            outputs = outputs.view(-1)

            loss = self.loss(embeddings, outputs, targets)

            self.log(f"{log}_loss", loss, prog_bar=True)
            self.log(f"{log}_mae", self.mae(outputs, targets), prog_bar=True)
            self.log(f"{log}_mse", self.mse(outputs, targets), prog_bar=True)
            self.log(f"{log}_r2", self.r2(outputs, targets), prog_bar=True)

            self.log(f"{log}_acc", 1 / (loss + 1e-8), prog_bar=False)

        if log_metrics:
            if self.n_classes > 1:
                self.log(f"{log}_ck", self.ck(outputs, targets))
                self.log(f"{log}_pr", self.pr(outputs, targets))
                self.log(f"{log}_rc", self.rc(outputs, targets))

        return loss

    def training_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs)

        return self.compute_loss(embeddings, outputs, targets)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs)

        return self.compute_loss(embeddings, outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch

        embeddings, outputs = self.encode(inputs)

        return self.compute_loss(embeddings, outputs, targets, "test", log_metrics=True)

####################################################################################################

class SleepAutoEncoderModuleOld(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SleepAutoEncoderModuleOld, self).__init__()
        self.nn = nn
        
        self.loss = config["loss_call"]
        #self.gaussian_kernel = self.gaussian_kernel(3)
        self.module_config = config


    def forward(self, x):
        return self.nn(x)


    def gaussian_kernel(self, size, sigma=0.6):
        """Create a 2D Gaussian kernel."""
        kernel = torch.tensor([[i, j] for i in range(size) for j in range(size)], dtype=torch.float32)
        mean = (size - 1) / 2.0
        variance = sigma**2
        
        # Calculate the Gaussian distribution
        kernel = torch.exp(-(torch.sum((kernel - mean)**2, dim=1)) / (2.0 * variance))
        kernel = kernel / torch.sum(kernel)
        
        # Reshape to a 2D matrix
        return kernel.reshape(size, size).unsqueeze(0).unsqueeze(0)

    def apply_smoothing(self, tensor: torch.Tensor):
        # Estrai le dimensioni del tensore
        batch_size, L, C, height, width = tensor.size()
        
        kernel = self.gaussian_kernel.to(tensor.device)
        
        # Per applicare la convoluzione, dobbiamo appiattire il tensore lungo la dimensione del batch e dei canali
        tensor = tensor.view(batch_size * L * C, 1, height, width)
        
        # Applica la convoluzione 2D con padding per mantenere la dimensione dell'immagine
        smoothed_tensor = Fun.conv2d(tensor, kernel, padding=kernel.size(3)//2)
        #print("smoothed_tensor:", smoothed_tensor.size())
        
        # Ripristina le dimensioni originali
        smoothed_tensor = smoothed_tensor.view(batch_size, L, C, height, width)
        #print("smoothed_tensor view:", smoothed_tensor.size())

        return smoothed_tensor

    def remove_wake_epochs(self, x, label):
        #print(label)
        mask = (label != 0).all(dim=1)
        #print(mask)
        x = x[mask]
        #print(x.size())
        return x

    def compute_loss(
        self,
        inputs,
        input_hat,
        log: str = "train",
    ):
        loss = self.loss(inputs, input_hat)
        std_pred_T = torch.std(input_hat, dim=(-2))
        std_input_T = torch.std(inputs, dim=(-2))
        std_pred_F = torch.std(input_hat, dim=(-1))
        std_input_F = torch.std(inputs, dim=(-1))
        std_input = torch.std(inputs, dim=(-2, -1))
        std_pred = torch.std(input_hat, dim=(-2, -1))
        #mean_pred = torch.std(input_hat, dim=(-2, -1))

        #std_penalty = torch.mean((1 / (std_pred + 1e-8))
        std_penalty_T = torch.mean((std_pred_T - std_input_T)**2)
        std_penalty_F = torch.mean((std_pred_F - std_input_F)**2)
        std_penalty = torch.mean((std_input - std_pred)**2)
        #mean_penalty = torch.mean((mean_pred)**2)
        loss_penalized = 2*loss + std_penalty + 0.5* std_penalty_T + 0.5 * std_penalty_F
        
        self.log(f"{log}_loss", loss, prog_bar=True)
        self.log(f"{log}_std_penalty", std_penalty, prog_bar=True)
        self.log(f"{log}_std_penalty_T", std_penalty_T, prog_bar=True)
        self.log(f"{log}_std_penalty_F", std_penalty_F, prog_bar=True)
        return loss_penalized
    
    def training_step(self, batch, batch_idx):
        # Logica di training
        x, labels = batch
        #x = self.remove_wake_epochs(x, labels)

        #x = self.apply_smoothing(x)
        x_hat = self.forward(x)
        return self.compute_loss(x, x_hat)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        x, labels = batch
        #x = self.remove_wake_epochs(x, labels)
        #x_smoothed = self.apply_smoothing(x)
        #x_hat = self.forward(x_smoothed)
        x_hat = self.forward(x)
        return self.compute_loss(x, x_hat, log="val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        x, labels = batch
        #x = self.remove_wake_epochs(x, labels)
        #x_smoothed = self.apply_smoothing(x)
        #x_hat = self.forward(x_smoothed)
        x_hat = self.forward(x)
        return self.compute_loss(x, x_hat, log="test")
    
    def configure_optimizers(self):
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=1e-4,
            weight_decay=1e-3,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="max",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )

        return {
            "optimizer": self.opt,
            "lr_scheduler": {"scheduler": self.scheduler, "monitor": "val_loss"},
        }


class SleepAutoEncoderModule(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SleepAutoEncoderModule, self).__init__()
        self.nn = nn
        self.module_config = config
        if(config["loss_name"] == "w_mse"):
            self.loss = config["loss_call"](config)
            self.std_penalty = Reconstruction().std_penalty_w
        else:
            self.loss = config["loss_call"]
            self.std_penalty = Reconstruction().std_penalty
        
    def forward(self, x):
        return self.nn(x)


    def compute_loss(
        self,
        inputs,
        input_hat,
        log: str = "train",
    ):

        if(self.module_config["loss_name"] == "w_mse"):
            mse_vanilla, mse_first_freq, mse_last_freq, mse = self.loss(inputs, input_hat)
            self.log(f"{log}_mse_vanilla", mse_vanilla, prog_bar=True)
            self.log(f"{log}_mse_first_freq", mse_first_freq, prog_bar=True)
            self.log(f"{log}_mse_last_freq", mse_last_freq, prog_bar=True)
        else:
            mse = self.loss(inputs, input_hat)
        
        std_penalty, std_penalty_T, std_penalty_F = self.std_penalty(inputs, input_hat)
        loss_tot = 2 * mse + 0.5 * std_penalty + 0.5 * std_penalty_T + 0.5 * std_penalty_F

        self.log(f"{log}_loss", mse, prog_bar=True)
        self.log(f"{log}_loss_tot", loss_tot, prog_bar=True)
        self.log(f"{log}_std_penalty", std_penalty, prog_bar=True)
        self.log(f"{log}_std_penalty_T", std_penalty_T, prog_bar=True)
        self.log(f"{log}_std_penalty_F", std_penalty_F, prog_bar=True)
        return loss_tot
        
    def training_step(self, batch, batch_idx):
        # Logica di training
        x, labels = batch
        x_hat = self.forward(x)
        return self.compute_loss(x, x_hat)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        x, labels = batch
        x_hat = self.forward(x)
        return self.compute_loss(x, x_hat, log="val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        x, labels = batch
        x_hat = self.forward(x)
        return self.compute_loss(x, x_hat, log="test")
    
    def configure_optimizers(self):
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=1e-4,
            weight_decay=1e-3,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="max",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )

        return {
            "optimizer": self.opt,
            "lr_scheduler": {"scheduler": self.scheduler, "monitor": "val_loss"},
        }