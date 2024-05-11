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
        super(SeqSleepNetCEM, self).__init__(Net(module_config), module_config)

        self.n_classes = module_config["n_classes"]
        self.n_concept = module_config["n_concept"]
        self.class_division = eval(module_config["class_division"]) if module_config.get("class_division") is not None else None
        #alpha is the parameter for the concept loss scaling in the end_to_end training
        self.alpha = float(module_config["alpha"])

        #defining loss and measures for concepts
        self.loss_cocept = CrossEntropyLoss(module_config["loss_params"])
        self.acc_concept = tm.Accuracy(
            task="multiclass", num_classes=self.n_concept#, average="weighted"
        )
        self.f1_concept = tm.F1Score(
            task="multiclass", num_classes=self.n_concept#, average="weighted"
        )
        self.ck_concept = tm.CohenKappa(task="multiclass", num_classes=self.n_concept)
        self.pr_concept = tm.Precision(
            task="multiclass", num_classes=self.n_concept#, average="weighted"
        )
        self.rc_concept = tm.Recall(
            task="multiclass", num_classes=self.n_concept#, average="weighted"
        )
        
        #defining loss and measures for classes
        self.loss = CrossEntropyLoss(module_config["loss_params"])
        self.acc_target = self.acc
        self.f1_target = self.f1
        self.ck_target = self.ck
        self.pr_target = self.pr
        self.rc_target = self.rc

    def compute_loss(
        self,
        concepts,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        #concepts[0] = concepts_embedding; concepts[1] = concepts_activations
        activations = concepts[1]
        activations = activations.reshape(-1, self.n_concept)
        targets = targets.reshape(-1)
        act_targets = targets
        loss_concept = self.loss_cocept(None, activations, act_targets)
        outputs = outputs.reshape(-1, self.n_classes)

        if self.n_classes > 2:
            class_targets = targets
        else:
            #Binary case
            if(self.class_division != None):
                class_targets = torch.tensor([0 if t in self.class_division[0] else 1 for t in targets]).to('cuda')
            else:
                print("class division is None")

        loss_target = self.loss(None, outputs, class_targets)
        self.log_function(log, log_metrics, loss_concept, loss_target, activations, outputs, act_targets, class_targets)

        return self.alpha * loss_concept + loss_target
    
    def log_function (self, log, log_metrics, loss_concept , loss_target, activations, outputs, act_targets, targets):
        self.log(f"{log}_loss_target", loss_target, prog_bar=True)
        self.log(f"{log}_loss_concept", loss_concept, prog_bar=True)
        self.log(f"{log}_acc", self.acc_target(outputs, targets))
        self.log(f"{log}_acc_concept", self.acc_concept(activations, act_targets))
        self.log(f"{log}_f1", self.f1_target(outputs, targets))
        self.log(f"{log}_f1_concept", self.f1_concept(activations, act_targets))
        if log_metrics:
            None


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

        x = self.sequence_encoder_cem.encode(x)

        x = self.sequence_encoder_cem(x)

        y = self.sequence_encoder_cem.clf(x)

        return x, y

    def forward(self, x):
        x, y = self.encode(x)
        return y


class SeqSleepNetCEM_old(SleepModule):
    def __init__(self, module_config=module_config):
        super(SeqSleepNetCEM, self).__init__(
            EpochEncoder(module_config),
            SequenceEncoderCEM(module_config),
            module_config,
        )
        self.n_classes = module_config["n_classes"]
        self.n_concept = module_config["n_concept"]
        self.class_division = eval(module_config["class_division"]) if module_config.get("class_division") is not None else None
        self.training_type = module_config["training_type"]
        #alpha is the parameter for the concept loss scaling in the end_to_end training
        self.alpha = float(module_config["alpha"])
        #optimizer parameters
        self.lr = float(module_config["learning_rate"])
        self.adam_b_1 = float(module_config["adam_beta_1"])
        self.adam_b_2 = float(module_config["adam_beta_2"])
        self.adam_e = float(module_config["adam_epsilon"])

        if self.training_type == "modular_1":
            self.automatic_optimization = False

        module_config["loss_params"]["class_weights"] = None

        self.loss_cocept = CrossEntropyLoss(module_config["loss_params"])
        self.acc_concept = tm.Accuracy(
            task="multiclass", num_classes=self.n_concept#, average="weighted"
        )
        self.f1_concept = tm.F1Score(
            task="multiclass", num_classes=self.n_concept#, average="weighted"
        )
        self.ck_concept = tm.CohenKappa(task="multiclass", num_classes=self.n_concept)
        self.pr_concept = tm.Precision(
            task="multiclass", num_classes=self.n_concept#, average="weighted"
        )
        self.rc_concept = tm.Recall(
            task="multiclass", num_classes=self.n_concept#, average="weighted"
        )
        
        self.loss = CrossEntropyLoss(module_config["loss_params"])
        self.acc_target = self.acc
        self.f1_target = self.f1
        self.ck_target = self.ck
        self.pr_target = self.pr
        self.rc_target = self.rc
        

    

    def training_step(self, batch, batch_idx):
        if(self.training_type == "end_to_end"):
            return super().training_step(batch, batch_idx)
        elif self.training_type == "modular_1":
            inputs, targets = batch
            concepts, outputs = self.encode(inputs)
            opt_cem, opt = self.optimizers()

            loss_cem, loss_target = self.compute_loss(concepts, outputs, targets)

            opt_cem.zero_grad()
            self.manual_backward(loss_cem, retain_graph=True)
            opt.zero_grad()
            self.manual_backward(loss_target)

            opt_cem.step()
            opt.step()
    
    def configure_optimizers(self):
        if(self.training_type == "end_to_end"):
            return super().configure_optimizers()
        elif self.training_type == "modular_1":
            params = list(self.nn.epoch_encoder.parameters()) + \
                    list(self.nn.sequence_encoder.LSTM.parameters()) + \
                    list(self.nn.sequence_encoder.proj.parameters()) + \
                    list(self.nn.sequence_encoder.norm.parameters()) + \
                    list(self.nn.sequence_encoder.cls.parameters())
                    
            params_cem = list(self.nn.sequence_encoder.cem.parameters())

            optimizer_cem = torch.optim.Adam(params_cem,
                                             lr=self.lr*10,betas=(self.adam_b_1,self.adam_b_2,),eps=self.adam_e,
                                             )
            optimizer = torch.optim.Adam(params,
                                         lr=self.lr,betas=(self.adam_b_1,self.adam_b_2,),eps=self.adam_e,
                                         )
            
            return optimizer_cem, optimizer
    
    
            # self.log(f"{log}_ck", self.ck_target(outputs, targets))
            # self.log(f"{log}_pr", self.pr_target(outputs, targets))
            # self.log(f"{log}_pr_concept", self.pr_concept(activations, act_targets))
            # self.log(f"{log}_rc", self.rc_target(outputs, targets))
            # self.log(f"{log}_rc_concept", self.rc_concept(activations, act_targets))   
                 

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
