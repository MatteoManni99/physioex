from typing import Dict

import torch.nn as nn
import torch.optim as optim
from physioex.train.networks.base import SleepModule


class SelfSupervisedSleepModule(SleepModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SelfSupervisedSleepModule, self).__init__(nn, config)
  
        # loss and loss_factor_names have to be defined in the child class
        # self.loss =               the loss function to be used
        # self.loss_factor_names =  a list of names of the loss factors to report in the logs [loss1, loss2, ...]
        
        # metrics and metric_names (if any) have to be defined in the child class
        # self.metrics =            a list of metrics to be used (if any) [metric1, metric2, ...]
        # self.metric_names =       a list of names of the metrics (if any) to report in the logs [m_name1, m_name2, ...]
        

        self.metrics = None
        self.metric_names = None
        
        # remove the inheritated metrics and loss from instance variables
        if self.n_classes > 1:
            del self.loss
            del self.wacc
            del self.macc
            del self.wf1
            del self.mf1
            del self.ck
            del self.pr
            del self.rc
        elif self.n_classes == 1:
            del self.mse
            del self.mae
            del self.r2


    def forward(self, x):
        return self.nn(x)

    def compute_loss(
        self,
        embeddings,
        inputs,
        rec_inputs, #reconstructed inputs
        log: str = "train",
        log_metrics: bool = False,
    ):
        #the first element of the loss_list is the total loss to be backpropagated
        loss_list = self.loss(embeddings, rec_inputs, inputs) #loss_list = [loss, loss1, loss2, ...] where loss is the total loss

        #log all loss factors in the loss_list
        for i, factor in enumerate(loss_list):
            self.log(f"{log}_{self.factor_names[i]}", factor, prog_bar=True)
        
        #log the metrics if log_metrics is True and the metrics are defined 
        if log_metrics and self.metrics is not None:
            for i, metric in enumerate(self.metrics):
                self.log(f"{log}_{self.metric_names[i]}", metric(inputs, rec_inputs), prog_bar=True)
        
        return loss_list[0]
    
    def training_step(self, batch):
        x, labels = batch #labels are not needed
        z, x_hat = self.forward(x)
        return self.compute_loss(z, x, x_hat)

    def validation_step(self, batch):
        x, labels = batch #labels are not needed
        z, x_hat = self.forward(x)
        return self.compute_loss(z, x, x_hat, log="val", log_metrics=True)
    
    def test_step(self, batch):
        x, labels = batch #labels are not needed
        z, x_hat = self.forward(x)
        return self.compute_loss(z, x, x_hat, log="test", log_metrics=True)