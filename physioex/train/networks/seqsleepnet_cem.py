from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torchmetrics as tm
from physioex.train.networks.utils.loss import CrossEntropyLoss

from physioex.train.networks.base import SleepModule
from physioex.train.networks.seqsleepnet import EpochEncoder, SequenceEncoder

module_config = dict()

class SeqSleepNetCEM(SleepModule):
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

                "n_concept": 15,
                "concept_dim": 2,
                "alpha": 20,
            }
        )

        super(SeqSleepNetCEM, self).__init__(Net(module_config), module_config)

        self.n_classes = module_config["n_classes"]
        self.n_concept = module_config["n_concept"]
        self.alpha = float(module_config["alpha"])
        self.central_epoch = int((module_config["sequence_length"] - 1) / 2)
        #defining loss and measures for classes
        self.loss_class = CrossEntropyLoss()
        self.acc_class = tm.Accuracy(task="multiclass", num_classes=self.n_classes)
        self.wf1_class = tm.F1Score(task="multiclass", num_classes=self.n_classes, average="weighted")
        self.mf1_class = tm.F1Score(task="multiclass", num_classes=self.n_classes, average="macro")
        self.ck_class = tm.CohenKappa(task="multiclass", num_classes=self.n_classes)
        self.pr_class = tm.Precision(task="multiclass", num_classes=self.n_classes)
        self.rc_class = tm.Recall(task="multiclass", num_classes=self.n_classes)

        del self.wacc
        del self.macc
        del self.wf1
        del self.mf1
        del self.ck
        del self.pr
        del self.rc
        del self.loss
        
        #defining loss and measures for concepts
        self.mse = nn.MSELoss()
        self.loss_concept = nn.L1Loss()
        

    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        epoch_emb, concept_emb = embeddings
        outputs_concept, outputs_class = outputs
        targets_class, targets_concept = targets

        outputs_concept = outputs_concept[:, self.central_epoch, :].squeeze()
        targets_concept = targets_concept[:, self.central_epoch, :]

        outputs_class = outputs_class.reshape(-1, self.n_classes)
        targets_class = targets_class.reshape(-1)
        loss_class = self.loss_class(None, outputs_class, targets_class)

        loss_concept = self.loss_concept(outputs_concept, targets_concept)
        loss = loss_class + self.alpha * loss_concept

        self.log(f"{log}_loss_class", loss_class, prog_bar=True)
        self.log(f"{log}_acc", self.acc_class(outputs_class, targets_class), prog_bar=True)
        self.log(f"{log}_mf1", self.mf1_class(outputs_class, targets_class), prog_bar=True)
        
        self.log(f"{log}_loss_concept", loss_concept, prog_bar=True)
        self.log(f"{log}_mse_concept", self.mse(outputs_concept, targets_concept), prog_bar=True)

        self.log(f"{log}_loss", loss, prog_bar=True)

        if(log_metrics):
            self.log(f"{log}_ck", self.ck_class(outputs_class, targets_class), prog_bar=True)
            self.log(f"{log}_pr", self.pr_class(outputs_class, targets_class), prog_bar=True)
            self.log(f"{log}_rc", self.rc_class(outputs_class, targets_class), prog_bar=True)

        return loss


class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.epoch_encoder = EpochEncoder(module_config)
        self.sequence_encoder_cem = SequenceEncoderCEM(module_config)

    def encode(self, x):
        batch, L, nchan, T, F = x.size()
        x = x.reshape(-1, nchan, T, F)
        x = self.epoch_encoder(x)
        x = x.reshape(batch, L, -1)
        epoch_emb, concept_emb, concept_act = self.sequence_encoder_cem.encode(x)
        y = self.sequence_encoder_cem.clf(concept_emb)

        return (epoch_emb, concept_emb), (concept_act, y)

    def forward(self, x):
        x, y = self.encode(x)
        return y

class SequenceEncoderCEM(SequenceEncoder):
    def __init__(self, module_config):
        super(SequenceEncoderCEM, self).__init__(module_config)
        self.concept_dim = module_config["concept_dim"]
        self.cem_input_dim = module_config["seqnhidden2"] * 2
        self.n_classes = module_config["n_classes"]
        self.n_concept = module_config["n_concept"]
        self.cem = CEM(self.cem_input_dim, self.n_concept, self.concept_dim)
        self.clf = nn.Linear(self.n_concept * self.concept_dim, self.n_classes)
    

    def forward(self, x):
        x, _ = self.encode(x)
        x = self.clf(x)
        return x

    def encode(self, x):
        x = super().encode(x)
        concept_emb, activations = self.cem(x)
        return x, concept_emb, activations


class CEM(nn.Module):
    def __init__(self, input_dim, n_concept, concept_dim):
        super().__init__()
        self.n_concept = n_concept
        self.input2candidateConcepts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, concept_dim), nn.LeakyReLU())
                for _ in range(self.n_concept * 2)
            ]
        )

        self.score_function = nn.Sequential(
            nn.Linear(concept_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        concepts = []
        concept_activations = []
        for i in range(self.n_concept):
            c_plus = self.input2candidateConcepts[i](x)
            c_minus = self.input2candidateConcepts[i + 1](x)
            c = torch.cat((c_plus, c_minus), 2)
            score = self.score_function(c)
            concept = score * c_plus + (1 - score) * c_minus
            concepts.append(concept)
            concept_activations.append(score)
        return torch.cat(concepts, 2), torch.stack(concept_activations, 2)
