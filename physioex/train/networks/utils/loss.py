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
    def __init__(self):
        super(SimilarityCombinedLoss, self).__init__()
        self.miner = miners.MultiSimilarityMiner()
        self.contr_loss = losses.TripletMarginLoss(
            distance=CosineSimilarity(),
            embedding_regularizer=LpRegularizer(),
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, preds, targets):
        loss = self.ce_loss(preds, targets)
        hard_pairs = self.miner(emb, targets)

        return loss + self.contr_loss(emb, targets, hard_pairs)


class CrossEntropyLoss(nn.Module, PhysioExLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, preds, targets):
        return self.ce_loss(preds, targets)

class ReconstructionPenalties(nn.Module, PhysioExLoss):
    def __init__(self, weighted = False, split = 50):
        super(ReconstructionPenalties, self).__init__()
        self.weighted = weighted
        self.split = split

    def std_penalty(self, preds, targets):
        std_pred = torch.std(preds, dim=(-2, -1))
        std_target = torch.std(targets, dim=(-2, -1))
        std_pred_T = torch.std(preds, dim=(-2))
        std_target_T = torch.std(targets, dim=(-2))
        std_pred_F = torch.std(preds, dim=(-1))
        std_target_F = torch.std(targets, dim=(-1))

        std_penalty = torch.mean((std_pred - std_target)**2)
        std_penalty_T = torch.mean((std_pred_T - std_target_T)**2)
        std_penalty_F = torch.mean((std_pred_F - std_target_F)**2)

        return std_penalty, std_penalty_T, std_penalty_F

    def w_std_penalty(self, preds, targets, split = 50):
        preds = preds[..., :split]
        targets = targets[..., :split]

        return self.std_penalty(preds, targets)

    def forward(self, preds, targets):
        if self.weighted:
            return self.w_std_penalty(preds, targets, self.split)
        else:
            return self.std_penalty(preds, targets)

class ReconstructionLoss(nn.Module, PhysioExLoss):
    def __init__(self, alpha1, alpha2, alpha3, alpha4):
        super(ReconstructionLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.rec_penalties = ReconstructionPenalties()
        self.mse = nn.MSELoss()
        
    def forward(self, embeddings, preds, targets):
        std_pen, std_pen_T, std_pen_F = self.rec_penalties(preds, targets)
        mse = self.mse(preds, targets)
        loss = mse * self.alpha1 + std_pen * self.alpha2 + std_pen_T * self.alpha3 + std_pen_F * self.alpha4
        return loss, mse, std_pen, std_pen_T, std_pen_F


class SemiSupervisedLoss(nn.Module, PhysioExLoss):
    def __init__(self, alpha1, alpha2, alpha3, alpha4, lambda1, lambda2, lambda3, lambda4):
        super(SemiSupervisedLoss, self).__init__()
        self.recostruc_loss = ReconstructionLoss(alpha1, alpha2, alpha3, alpha4)
        self.cel = CrossEntropyLoss()
        self.l1 = lambda1
        self.l2 = lambda2
        self.l3 = lambda3
        self.l4 = lambda4
        
    def forward(self, emb, proto, preds, targets, inputs, inputs_hat):
        rec_loss, mse, std_pen, std_pen_T, std_pen_F = self.recostruc_loss(None, inputs_hat, inputs)
        cel = self.cel(None, preds, targets)
        r1 = torch.mean(torch.min(torch.cdist(proto, emb), dim=1).values)
        r2 = torch.mean(torch.min(torch.cdist(emb, proto), dim=1).values)
        loss = self.l1 * cel + self.l2 * rec_loss + self.l3 * r1 + self.l4 * r2

        return loss, cel, r1, r2, rec_loss, mse, std_pen, std_pen_T, std_pen_F

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

class HuberLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

        self.loss = nn.HuberLoss(delta=5)

    def forward(self, emb, preds, targets):
        return self.loss(preds, targets) / 112.5


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

        # mse
        self.loss = nn.MSELoss()

    def forward(self, emb, preds, targets):
        return self.loss(preds, targets)


config = {"cel": CrossEntropyLoss, "scl": SimilarityCombinedLoss, "reg": RegressionLoss, "mse": nn.MSELoss(reduction="mean"), "w_mse": WeightedMSELoss}
