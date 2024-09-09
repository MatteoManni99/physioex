from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from physioex.train.networks.base import SleepWrapperModule
from physioex.train.networks.seqsleepnet import SeqSleepNet


module_config = dict()

class Wrapper(SleepWrapperModule):
    def __init__(self, module_config=module_config):
        super(Wrapper, self).__init__(Net(module_config), module_config)


class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.load_and_freeze_model(module_config)
        
        self.n_proto_per_class = module_config["n_proto_per_class"]
        self.n_classes = module_config["n_classes"]
        self.n_prototypes = self.n_proto_per_class * self.n_classes

        self.L = int(module_config["seq_len"])
        self.T = int(module_config["T"])
        self.F = int(module_config["F"])
        self.nchan = int(module_config["in_channels"])
       
        self.prototypes = nn.Parameter(torch.rand(self.n_prototypes, self.nchan, self.T, self.F))
        #self.prototypes debug version
        # self.prototypes = torch.randn(self.n_prototypes, self.nchan, self.T, self.F)
        # for i in range(self.n_prototypes):
        #     self.prototypes[i] = nn.Parameter(torch.ones(self.nchan, self.T, self.F)*i)
        #self.central_epoch = int((self.L - 1) / 2)


    def encode(self, inputs, labels):
        #print("PROTOTYPES:", self.prototypes.size())
        batch_size = labels.size(0)
        proto_batch = torch.zeros(batch_size, self.L, self.nchan, self.T, self.F).to(labels.device)

        for batch_idx, label_epochs in enumerate(labels):
            #offset to pick up the right prototype
            offset = batch_idx%self.n_proto_per_class*self.n_classes
            for epoch_idx, label in enumerate(label_epochs):
                proto_batch[batch_idx][epoch_idx] = self.prototypes[label + offset]

        emb, pred = self.wrapped_model.encode(proto_batch)

        # proto_to_embed = torch.zeros(self.n_prototypes, self.L, self.nchan, self.T, self.F).to(labels.device)
        # for i in range(self.n_prototypes):
        #     proto_to_embed[i] = self.prototypes[i].repeat(3, 1, 1, 1)
        # proto_emb, _ = self.wrapped_model.encode(proto_to_embed)

        proto_to_embed = self.prototypes.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        proto_emb, _ = self.wrapped_model.encode(proto_to_embed)

        return (emb, proto_emb) , pred

    def forward(self, inputs, labels):
        emb, pred = self.encode(inputs, labels)
        return pred
    
    def load_and_freeze_model(self, module_config):
        # self.wrapped_model = load_pretrained_model(
        #     name=module_config["model_name"],
        #     in_channels=module_config["in_channels"],
        #     sequence_length=module_config["seq_len"],
        #     softmax=False,
        #     loss = "cel",
        #     ckpt_path=module_config["ckpt_path"],
        # ).eval()
        self.wrapped_model = SeqSleepNet.load_from_checkpoint(
            checkpoint_path = module_config["model_path"],
            module_config=module_config
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(self.wrapped_model.device)
        for param in self.wrapped_model.parameters():
            param.requires_grad = False