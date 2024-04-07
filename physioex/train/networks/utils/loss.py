from abc import ABC, abstractmethod
from typing import Dict

import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ClassWeightedReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from torch import nn


class PhysioExLoss(ABC):

    @abstractmethod
    def forward(self, emb, preds, targets):
        pass


class SimilarityCombinedLoss(nn.Module, PhysioExLoss):
    def __init__(self, params: Dict):
        super(SimilarityCombinedLoss, self).__init__()
        self.miner = miners.MultiSimilarityMiner()
        self.contr_loss = losses.TripletMarginLoss(
            distance=CosineSimilarity(),
            reducer=ClassWeightedReducer(weights=params["class_weights"]),
            embedding_regularizer=LpRegularizer(),
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, preds, targets):
        loss = self.ce_loss(preds, targets)
        hard_pairs = self.miner(emb, targets)

        return loss + self.contr_loss(emb, targets, hard_pairs)


class CrossEntropyLoss(nn.Module, PhysioExLoss):
    def __init__(self, params: Dict = None):
        super(CrossEntropyLoss, self).__init__()

        # check if class weights are provided in params
        weights = params.get("class_weights") if params is not None else None
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, emb, preds, targets):
        return self.ce_loss(preds, targets)


class CrossEntropyLossCEM(nn.Module, PhysioExLoss):
    # TODO implement the case in which concepts are not equal to classes
    def __init__(self, params: Dict = None):
        super(CrossEntropyLossCEM, self).__init__()

        # check if class weights are provided in params
        weights = params.get("class_weights") if params is not None else None
        self.ce_loss_target = nn.CrossEntropyLoss(weight=weights)
        self.ce_loss_concepts = nn.CrossEntropyLoss()

    def forward(self, activations, preds, targets):
        loss_target = self.ce_loss_target(preds, targets)
        loss_concepts = self.ce_loss_concepts(activations, targets)
        return loss_target + loss_concepts


class BCELossCEM(nn.Module, PhysioExLoss):
    def __init__(self, params: Dict = None):
        super(BCELossCEM, self).__init__()
        self.class_division = ((0,), (1, 2, 3, 4))
        # check if class weights are provided in params and in case compute the weights for the two classes
        if params is not None:
            self.weights = torch.zeros(2, requires_grad=False)
            base_weights = params.get("class_weights")
            if self.class_division is None:
                self.weights = base_weights
            else:
                base_weights = 1/base_weights
                for i in self.class_division[0]:
                    self.weights[0] += base_weights[i]
                for i in self.class_division[1]:
                    self.weights[1] += base_weights[i]
                self.weights = 1/self.weights
                self.weights = self.weights / self.weights.sum()
        else:
            self.weights = None

        self.ce_loss_concepts = nn.CrossEntropyLoss()

    def forward(self, activations, preds, targets):
        binary_targets = targets[0]
        concept_targets = targets[1]
    
        if (self.weights is not None):
            weights = torch.where(binary_targets == 0, self.weights[0], self.weights[1]).to('cuda')
        else:
            weights = None

        bce_loss = nn.BCELoss(weight=weights)
        loss_target = bce_loss(preds, binary_targets)
        loss_concepts = self.ce_loss_concepts(activations, concept_targets)

        return loss_target + loss_concepts


config = {
    "cel": CrossEntropyLoss,
    "scl": SimilarityCombinedLoss,
    "cel_cem": CrossEntropyLossCEM,
    "bcel_cem": BCELossCEM,
}
