import importlib
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
from physioex.train.networks.utils.loss import Reconstruction, CrossEntropyLoss


class SleepModule(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SleepModule, self).__init__()
        self.save_hyperparameters(ignore=["nn"])
        self.nn = nn

        self.n_classes = config["n_classes"]

        if self.n_classes > 1:
            # classification experiment
            self.wacc = tm.Accuracy(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
            self.macc = tm.Accuracy(
                task="multiclass", num_classes=config["n_classes"], average="macro"
            )
            self.wf1 = tm.F1Score(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
            self.mf1 = tm.F1Score(
                task="multiclass", num_classes=config["n_classes"], average="macro"
            )
            self.ck = tm.CohenKappa(task="multiclass", num_classes=config["n_classes"])
            self.pr = tm.Precision(
                task="multiclass", num_classes=config["n_classes"]#, average="weighted"
            )
            self.rc = tm.Recall(
                task="multiclass", num_classes=config["n_classes"]#, average="weighted"
            )
        elif self.n_classes == 1:
            # regression experiment
            self.mse = tm.MeanSquaredError()
            self.mae = tm.MeanAbsoluteError()
            self.r2 = tm.R2Score()

        # loss
        loss_module, loss_class = config["loss"].split(":")
        self.loss = getattr(importlib.import_module(loss_module), loss_class)(
            **config["loss_kwargs"]
        )
        self.module_config = config

        # learning rate

        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            # verbose=True,
        )
        scheduler = {
            "scheduler": scheduler,
            "name": "lr_scheduler",
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [self.opt], [scheduler]

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

        batch_size, seq_len, n_class = outputs.size()

        embeddings = embeddings.reshape(batch_size * seq_len, -1)
        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)

        if self.n_classes > 1:
            loss = self.loss(embeddings, outputs, targets)

            self.log(f"{log}_loss", loss, prog_bar=True)
            self.log(f"{log}_acc", self.wacc(outputs, targets), prog_bar=True)
            self.log(f"{log}_f1", self.wf1(outputs, targets), prog_bar=True)
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
                self.log(f"{log}_macc", self.macc(outputs, targets))
                self.log(f"{log}_mf1", self.mf1(outputs, targets))
        return loss

    def training_step(self, batch, batch_idx):
        # get the logged metrics
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))

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
        
        self.alpha1 = config["alpha1"]
        self.alpha2 = config["alpha2"]
        self.alpha3 = config["alpha3"]
        self.alpha4 = config["alpha4"]
        
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
        loss = self.alpha1*mse + self.alpha2*std_penalty + self.alpha3*std_penalty_T + self.alpha4*std_penalty_F

        self.log(f"{log}_mse", mse, prog_bar=True)
        self.log(f"{log}_loss", loss, prog_bar=True)
        self.log(f"{log}_std_pen", std_penalty, prog_bar=True)
        self.log(f"{log}_std_pen_T", std_penalty_T, prog_bar=True)
        self.log(f"{log}_std_pen_F", std_penalty_F, prog_bar=True)
        return loss
        
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
    
class SleepAEPrototypeModule(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SleepAEPrototypeModule, self).__init__()
        self.nn = nn
        self.module_config = config
        self.central_epoch = int((config["seq_len"] - 1) / 2)
        self.mse = config["loss_call"]
        self.std_penalty = Reconstruction().std_penalty
        self.cel = CrossEntropyLoss()
        self.l1 = config["lambda1"]
        self.l2 = config["lambda2"]
        self.l3 = config["lambda3"]
        self.l4 = config["lambda4"]
        #self.l5 = config["lambda5"]
        self.f1 = tm.F1Score(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
        self.mf1 = tm.F1Score(
            task="multiclass", num_classes=config["n_classes"], average="macro"
        )
    def forward(self, x):
        return self.nn(x)

    def compute_loss(
        self,
        inputs,
        inputs_hat,
        embeddings,
        labels,
        pred,
        log: str = "train",
    ):
        inputs = inputs[:, self.central_epoch, 0, :, :]
        inputs_hat = inputs_hat[:, self.central_epoch, 0, :, :]
        embeddings = embeddings[:, self.central_epoch, :]
        labels = labels[:, self.central_epoch]
        
        cel = self.cel(None, pred, labels)
        
        mse = self.mse(inputs, inputs_hat)
        std_penalty, std_penalty_T, std_penalty_F = self.std_penalty(inputs, inputs_hat)
        reconstruction_loss = 1 * mse + 0.1 * std_penalty + 0.3 * std_penalty_T + 0.1 * std_penalty_F

        r1 = torch.mean(torch.min(torch.cdist(self.nn.classifier.prototypes, embeddings), dim=1).values)
        r2 = torch.mean(torch.min(torch.cdist(embeddings, self.nn.classifier.prototypes), dim=1).values)
        #----------------------------------
        # min_distances, closest_prototypes = torch.min(torch.cdist(embeddings, self.nn.classifier.prototypes), dim=1)
        # prototype_counts = torch.bincount(closest_prototypes, minlength=self.nn.classifier.prototypes.size(0))
        # mean_count = prototype_counts.float().mean()
        # abs_std = (abs(prototype_counts.float() - mean_count)).mean()
        # r2 = torch.mean(min_distances)
        #r3 = abs_std
        #----------------------------------
        #r3
        # proto_distances = torch.cdist(self.nn.classifier.prototypes, self.nn.classifier.prototypes)
        # mask = torch.eye(proto_distances.size(0), device=proto_distances.device)
        # proto_distances = proto_distances.masked_fill(mask.bool(), float('inf'))
        # min_distances, _ = torch.min(proto_distances, dim=1)
        # log_min_distances = torch.log(min_distances + 1 + 1e-8)
        # r3 = - torch.mean(log_min_distances)
    
        tot_loss = self.l1 * cel + self.l2 * reconstruction_loss + self.l3 * r1 + self.l4 * r2 #+ self.l5 * r3

        self.log(f"{log}_loss", tot_loss, prog_bar=True)
        self.log(f"{log}_cel", cel, prog_bar=True)
        self.log(f"{log}_f1", self.f1(pred, labels), prog_bar=True)
        self.log(f"{log}_mf1", self.mf1(pred, labels), prog_bar=True)
        self.log(f"{log}_r1", r1, prog_bar=True)
        self.log(f"{log}_r2", r2, prog_bar=True)
        #self.log(f"{log}_r3", r3, prog_bar=True)
        self.log(f"{log}_reconstr_loss", reconstruction_loss, prog_bar=True)
        self.log(f"{log}_mse", mse, prog_bar=True)
        self.log(f"{log}_std_pen", std_penalty, prog_bar=True)
        self.log(f"{log}_std_pen_T", std_penalty_T, prog_bar=True)
        self.log(f"{log}_std_pen_F", std_penalty_F, prog_bar=True)
        return tot_loss
        
    def training_step(self, batch, batch_idx):
        # Logica di training
        x, labels = batch
        x_hat, embeddings, pred = self.forward(x)
        return self.compute_loss(x, x_hat, embeddings, labels, pred)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        x, labels = batch
        x_hat, embeddings, pred = self.forward(x)
        return self.compute_loss(x, x_hat, embeddings, labels, pred, log="val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        x, labels = batch
        x_hat, embeddings, pred = self.forward(x)
        return self.compute_loss(x, x_hat, embeddings, labels, pred, log="test")
    
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

class SleepWrapperModule(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SleepWrapperModule, self).__init__()
        #self.save_hyperparameters()
        self.nn = nn

        self.n_classes = config["n_classes"]
        self.n_proto_per_class = config["n_proto_per_class"]
        self.triangular_number = (self.n_proto_per_class - 1) * (self.n_proto_per_class) / 2
        self.lambda1 = config["lambda1"]
        self.lambda2 = config["lambda2"]

        self.proto_indices = [[j * self.n_classes + i for j in range(self.n_proto_per_class)] for i in range(self.n_classes)]
        # classification experiment
        self.acc = tm.Accuracy(
            task="multiclass", num_classes=config["n_classes"], average="weighted"
        )
        self.wf1 = tm.F1Score(
            task="multiclass", num_classes=config["n_classes"], average="weighted"
        )
        self.mf1 = tm.F1Score(
            task="multiclass", num_classes=config["n_classes"], average="macro"
        )
        self.loss = config["loss_call"](config["loss_params"])
        self.module_config = config


    def forward(self, x, y):
        return self.nn(x, y)

    def encode(self, x, y):
        return self.nn.encode(x, y)

    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
    ):
        # print(targets.size())
        batch_size, seq_len, n_class = outputs.size()
        
        input_emb, proto_emb = embeddings

        input_emb = input_emb.reshape(batch_size * seq_len, -1)
        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)
        
        proto_self_sim = 0
        if self.n_proto_per_class >1:
            for idices in self.proto_indices:
                #s_emb = torch.unsqueeze(proto_emb[idices].view(1, 2, -1), 0)
                s_emb = proto_emb[idices].view(1, self.n_proto_per_class, -1)
                dist_matrix = torch.cdist(s_emb, s_emb, 2)
                proto_self_sim += 1/(torch.log((torch.triu(dist_matrix, diagonal=1).sum()/self.triangular_number) + 1))

        
        std_dev = torch.mean(torch.std(self.nn.prototypes, dim=(-2, -1)))
        std_loss = torch.abs(std_dev - 0.9)

        cel = self.loss(input_emb, outputs, targets)
        tot_loss = cel + self.lambda1 * proto_self_sim + self.lambda2 * std_loss

        self.log(f"{log}_loss", tot_loss, prog_bar=True)
        self.log(f"{log}_cel", cel, prog_bar=True)
        self.log(f"{log}_proto_self_sim", proto_self_sim, prog_bar=True)
        self.log(f"{log}_std_loss", std_loss, prog_bar=True)
        self.log(f"{log}_acc", self.acc(outputs, targets), prog_bar=True)
        self.log(f"{log}_wf1", self.wf1(outputs, targets), prog_bar=True)
        self.log(f"{log}_mf1", self.mf1(outputs, targets), prog_bar=True)

        return tot_loss

    def training_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs, targets)

        return self.compute_loss(embeddings, outputs, targets)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs, targets)

        return self.compute_loss(embeddings, outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch

        embeddings, outputs = self.encode(inputs, targets)

        return self.compute_loss(embeddings, outputs, targets, "test")
    
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
            "lr_scheduler": {"scheduler": self.scheduler, "monitor": "val_loss"},
        }