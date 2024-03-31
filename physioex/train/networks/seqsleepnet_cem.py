from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

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

        # embedding[0] = concepts_embedding ; embeddings[1] = concept_activations
        activations = embeddings[1]
        activations = activations.reshape(batch_size * seq_len, -1)
        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)

        # TODO fare l'assert della loss perché funziona solo con alcune loss
        loss = self.loss(activations, outputs, targets)

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
        self.n_concept = module_config["n_concepts"]
        self.embedding_dim = module_config["embedding_dim"]
        self.latent_dim = module_config["latent_space_dim"]

        self.cem = CEM(self.latent_dim, self.n_concept, self.embedding_dim)

        self.cls = nn.Linear(
            self.n_concept * self.embedding_dim, module_config["n_classes"]
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
    def __init__(self, input_dim, n_concept, embedding_dim):
        super().__init__()
        # self.concept_activations = concept_activations
        self.n_concept = n_concept
        # TODO aggiungere le altre attivazioni
        self.input2candidateConcepts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.LeakyReLU())
                for _ in range(self.n_concept * 2)
            ]
        )

        self.score_function = nn.Sequential(
            nn.Linear(embedding_dim * 2, 1),
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
