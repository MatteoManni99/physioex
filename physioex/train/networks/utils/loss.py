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

class WeightedMSELoss(nn.Module, PhysioExLoss):
    def __init__(self, params: Dict = None):
        super(WeightedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduce=False)
        self.border_frequency = 50
        self.weight_before = 2
        self.weight_after = 0
        self.batch_size = params.get("batch_size")
        self.weights = torch.ones(self.batch_size, params.get("seq_len"), params.get("in_channels"), params.get("T"), params.get("F"))
        self.weights[..., :self.border_frequency] *= self.weight_before
        self.weights[..., self.border_frequency:] *= self.weight_after

    def forward(self, preds, targets):
        mse_loss = self.mse_loss(preds, targets)
        if(preds.size(0) == self.batch_size):
            weighted_mse_loss = mse_loss * self.weights.to(preds.device)
        else:
            weighted_mse_loss = mse_loss * self.weights[:preds.size(0)].to(preds.device)

        mse_first_freq = mse_loss[..., :self.border_frequency]
        mse_last_freq = mse_loss[..., self.border_frequency:]
        return mse_loss.mean(), mse_first_freq.mean(), mse_last_freq.mean(), weighted_mse_loss.mean()

class RegressionLoss(nn.Module):
    def __init__(self, params: Dict = None):
        super(RegressionLoss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.params = params

    def forward(self, emb, preds, targets):
        mae = self.mae_loss(preds, targets)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        ss_res = torch.sum((targets - preds) ** 2)
        r2_score = 1 - ss_res / ss_tot

        combined_loss = 0.5 * mae + 0.5 * (1 - r2_score)
        return combined_loss


config = {"cel": CrossEntropyLoss, "scl": SimilarityCombinedLoss, "reg": RegressionLoss, "mse": nn.MSELoss(reduction="mean"), "w_mse": WeightedMSELoss}
