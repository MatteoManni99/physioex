from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torchmetrics as tm

from physioex.train.networks.base import SeqtoSeq
from physioex.train.networks.seqsleepnet import EpochEncoder, SequenceEncoder

module_config = dict()


class SeqSleepNetCEM(SeqtoSeq):
    def __init__(self, module_config=module_config):
        super(SeqSleepNetCEM, self).__init__(
            EpochEncoder(module_config),
            SequenceEncoderCEM(module_config),
            module_config,
        )
        self.class_division = eval(module_config["class_division"])

    def compute_loss(
        self,
        concepts,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        # print(targets.size())
        batch_size, seq_len, n_class = outputs.size()

        # concepts[0] = concepts_embedding ; concepts[1] = concept_activations
        activations = concepts[1]
        activations = activations.reshape(batch_size * seq_len, -1)
        
        #TODO sistemare i target; attualmente funziona solo se i concetti e le classi sono uguali
        targets = targets.reshape(-1)

        if n_class > 2:
            outputs = outputs.reshape(-1, n_class)
        else:
            #Binary case
            outputs = outputs.reshape(-1)
            if(self.class_division != None):
                binary_targets = torch.tensor([0 if t in self.class_division[0] else 1 for t in targets], dtype=torch.float32).to('cuda')
                concepts_targets = targets.reshape(-1)
                targets = (binary_targets, concepts_targets)
            else:
                print("class division is None") #TODO testare come entrare in questo caso cambiando il config

        # TODO fare l'assert della loss perché funziona solo con alcune loss
        # TODO raffinare la logica di questa funzione
        loss = self.loss(activations, outputs, targets)

        if n_class <= 2:
            targets = targets[0]
            self.acc = tm.Accuracy(task="binary").to("cuda")
            self.f1 = tm.F1Score(task="binary").to("cuda")
            self.ck = tm.CohenKappa(task="binary").to("cuda")
            self.pr = tm.Precision(task="binary").to("cuda")
            self.rc = tm.Recall(task="binary").to("cuda")
        
        #TODO questa parte potrebbe essere sposata in un'altra funzione (indipendente dal calcolo della loss) 
        self.log(f"{log}_loss", loss, prog_bar=True)
        self.log(f"{log}_acc", self.acc(outputs, targets), prog_bar=True)
        self.log(f"{log}_f1", self.f1(outputs, targets), prog_bar=True)

        if log_metrics:
            self.log(f"{log}_ck", self.ck(outputs, targets))
            self.log(f"{log}_pr", self.pr(outputs, targets))
            self.log(f"{log}_rc", self.rc(outputs, targets))

        return loss


class SequenceEncoderCEM(SequenceEncoder):
    def __init__(self, module_config):
        super(SequenceEncoderCEM, self).__init__(module_config)
        self.n_classes = module_config["n_classes"]
        self.n_concept = module_config["n_concepts"]
        self.concept_dim = module_config["concept_dim"]
        self.latent_dim = module_config["latent_space_dim"]

        self.cem = CEM(self.latent_dim, self.n_concept, self.concept_dim)

        if self.n_classes > 2:
            self.cls = nn.Linear(self.n_concept * self.concept_dim, self.n_classes)
        else:
            # binary case
            self.cls = nn.Sequential(
                nn.Linear(self.n_concept * self.concept_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x, _ = self.encode(x)
        x = self.cls(x)
        return x

    def encode(self, x):
        x = super().encode(x)
        embedding, activations = self.cem(x)
        return embedding, activations


class CEM(nn.Module):
    def __init__(self, input_dim, n_concept, concept_dim):
        super().__init__()
        # self.concept_activations = concept_activations
        self.n_concept = n_concept
        # TODO aggiungere le altre attivazioni
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
            # TODO volendo si potrebbe fare anche la concatenazione aggiungendo un'altra dimensione
            c = torch.cat((c_plus, c_minus), 2)
            score = self.score_function(c)
            concept = score * c_plus + (1 - score) * c_minus
            concepts.append(concept)
            concept_activations.append(score)
        return torch.cat(concepts, 2), torch.stack(concept_activations, 2)
