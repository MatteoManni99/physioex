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
            EpochEncoder(module_config), SequenceEncoderCEM(module_config), module_config
        )

class SequenceEncoderCEM(SequenceEncoder):
    def __init__(self, module_config):
        super(SequenceEncoderCEM, self).__init__(module_config)
        self.n_concept = module_config["n_concepts"]
        self.embedding_dim = module_config["embedding_dim"]
        self.latent_dim = module_config["latent_space_dim"]
        self.concept_activations = []
        
        self.cem = CEM(self.latent_dim, self.n_concept, self.embedding_dim, self.concept_activations)

        self.cls = nn.Linear(
            self.n_concept * self.embedding_dim, module_config["n_classes"]
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.cem(x)
        x = self.cls(x)
        #return x, self.concept_activations
        return x
    
class CEM(nn.Module):
    def __init__(self, input_dim, n_concept, embedding_dim, concept_activations):
        super().__init__()
        self.concept_activations = concept_activations
        self.n_concept = n_concept

        self.input2candidateConcepts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.LeakyReLU()) #TODO aggiungere le altre attivazioni
            for _ in range(self.n_concept*2)]
        )

        self.score_function = nn.Sequential(
            nn.Linear(embedding_dim*2, 1),
            nn.Sigmoid(),
        )
        #print(sum(p.numel() for p in self.parameters() if p.requires_grad))
        
    def forward(self, x):
        concepts = []
        self.concept_activations.clear()
        for i in range(self.n_concept):
            c_plus = self.input2candidateConcepts[i](x)
            c_minus = self.input2candidateConcepts[i+1](x)
            c = torch.cat((c_plus, c_minus), 2) #volendo si potrebbe fare anche la concatenazione aggiungendo un'altra dimensione
            score = self.score_function(c)
            concept = score * c_plus + (1-score) * c_minus
            concepts.append(concept)
            self.concept_activations.append(score) #to take trace of the concept activations for the loss function
        return torch.cat(concepts, 2)

