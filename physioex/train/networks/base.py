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
from physioex.train.networks.utils.loss import CrossEntropyLoss


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


class SelfSupervisedSleepModule(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SelfSupervisedSleepModule, self).__init__()
        self.save_hyperparameters(ignore=["nn"])
        self.nn = nn

        self.module_config = config

        # learning rate
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        
        # loss and metrics has to be defined in the child class

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

    def compute_loss(
        self,
        embeddings,
        inputs,
        rec_inputs, #reconstructed inputs
        log: str = "train",
        log_metrics: bool = False,
    ):
        #the first element of the loss_factors is the total loss
        loss_list = self.loss(embeddings, rec_inputs, inputs) #loss_list = [loss, loss1, loss2, ...] where loss is the total loss

        #log all loss factors in the loss_list
        for i, factor in enumerate(loss_list):
            self.log(f"{log}_{self.factor_names[i]}", factor, prog_bar=True)
        
        #log the metrics if log_metrics is True and the metrics are defined 
        if log_metrics and self.metrics is not None:
            for i, metric in enumerate(self.metrics):
                self.log(f"{log}_{self.metric_names[i]}", metric(inputs, rec_inputs), prog_bar=True)
        
        return loss_list[0]
    
    def training_step(self, batch):
        x, labels = batch #labels are not needed
        z, x_hat = self.forward(x)
        return self.compute_loss(z, x, x_hat)

    def validation_step(self, batch):
        x, labels = batch #labels are not needed
        z, x_hat = self.forward(x)
        return self.compute_loss(z, x, x_hat, log="val", log_metrics=True)
    
    def test_step(self, batch):
        x, labels = batch #labels are not needed
        z, x_hat = self.forward(x)
        return self.compute_loss(z, x, x_hat, log="test", log_metrics=True)


class SemiSupervisedSleepModuleModule(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SemiSupervisedSleepModuleModule, self).__init__()
        self.save_hyperparameters(ignore=["nn"])
        self.nn = nn

        self.module_config = config

        # learning rate
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        
        # loss and metrics has to be defined in the child class

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
            "lr_scheduler": {"scheduler": self.scheduler, "monitor": "val_loss"},
        }