from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from physioex.train.networks.base import SleepModule
from physioex.train.networks.seqsleepnet import SeqSleepNet
import torchmetrics as tm

module_config = dict()

class Wrapper(SleepModule):
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

                "n_classes": 5,
                "n_proto_per_class": 3,
                "lambda1": 0.04,
                "lambda2": 20,
                "model_path": "/home/manni/models/seqsleepnet_mass_l3_eeg_to_wrap/fold=-1-epoch=7-step=12872-val_loss=0.63.ckpt",
            }
        )
        super(Wrapper, self).__init__(Net(module_config), module_config)
        self.n_classes = module_config["n_classes"]
        self.n_proto_per_class = module_config["n_proto_per_class"]
        self.triangular_number = (self.n_proto_per_class - 1) * (self.n_proto_per_class) / 2
        self.lambda1 = module_config["lambda1"]
        self.lambda2 = module_config["lambda2"]

        self.proto_indices = [[j * self.n_classes + i for j in range(self.n_proto_per_class)] for i in range(self.n_classes)]

        self.acc = tm.Accuracy(task="multiclass", num_classes=module_config["n_classes"], average="weighted")
        self.wf1 = tm.F1Score(task="multiclass", num_classes=module_config["n_classes"], average="weighted")
        self.Mf1 = tm.F1Score(task="multiclass", num_classes=module_config["n_classes"], average="macro")

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
        batch_size, seq_len, n_class = outputs.size()
        
        input_emb, proto_emb = embeddings

        input_emb = input_emb.reshape(batch_size * seq_len, -1)
        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)
        
        proto_self_sim = 0
        if self.n_proto_per_class >1:
            for idices in self.proto_indices:
                s_emb = proto_emb[idices].view(1, self.n_proto_per_class, -1)
                dist_matrix = torch.cdist(s_emb, s_emb, 2)
                proto_self_sim += 1/(torch.log((torch.triu(dist_matrix, diagonal=1).sum()/self.triangular_number) + 1))

        
        std_dev = torch.mean(torch.std(self.nn.prototypes, dim=(-2, -1)))
        std_loss = torch.abs(std_dev - 0.9)

        cel = self.loss(None, outputs, targets)
        tot_loss = cel + self.lambda1 * proto_self_sim + self.lambda2 * std_loss

        self.log(f"{log}_loss", tot_loss, prog_bar=True)
        self.log(f"{log}_cel", cel, prog_bar=True)
        self.log(f"{log}_proto_self_sim", proto_self_sim, prog_bar=True)
        self.log(f"{log}_std_loss", std_loss, prog_bar=True)
        self.log(f"{log}_acc", self.acc(outputs, targets), prog_bar=True)
        self.log(f"{log}_wf1", self.wf1(outputs, targets), prog_bar=True)
        self.log(f"{log}_Mf1", self.Mf1(outputs, targets), prog_bar=True)

        return tot_loss
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs, targets)
        return self.compute_loss(embeddings, outputs, targets)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs, targets)
        return self.compute_loss(embeddings, outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs, targets)
        return self.compute_loss(embeddings, outputs, targets, "test")
    
class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.load_and_freeze_model(module_config)
        
        self.n_proto_per_class = module_config["n_proto_per_class"]
        self.n_classes = module_config["n_classes"]
        self.n_prototypes = self.n_proto_per_class * self.n_classes

        self.L = int(module_config["sequence_length"])
        self.T = int(module_config["T"])
        self.F = int(module_config["F"])
        self.nchan = int(module_config["in_channels"])
       
        self.prototypes = nn.Parameter(torch.rand(self.n_prototypes, self.nchan, self.T, self.F))


    def encode(self, inputs, labels):
        batch_size = labels.size(0)
        proto_batch = torch.zeros(batch_size, self.L, self.nchan, self.T, self.F).to(labels.device)

        for batch_idx, label_epochs in enumerate(labels):
            #offset to pick up the right prototype
            offset = batch_idx%self.n_proto_per_class*self.n_classes
            for epoch_idx, label in enumerate(label_epochs):
                proto_batch[batch_idx][epoch_idx] = self.prototypes[label + offset]

        emb, pred = self.wrapped_model.encode(proto_batch)

        proto_to_embed = self.prototypes.unsqueeze(1).repeat(1, self.L, 1, 1, 1)
        proto_emb, _ = self.wrapped_model.encode(proto_to_embed)

        return (emb, proto_emb) , pred

    def forward(self, inputs, labels):
        emb, pred = self.encode(inputs, labels)
        return pred
    
    def load_and_freeze_model(self, module_config):
        self.wrapped_model = SeqSleepNet.load_from_checkpoint(
            checkpoint_path = module_config["model_path"],
            module_config=module_config
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(self.wrapped_model.device)
        for param in self.wrapped_model.parameters():
            param.requires_grad = False