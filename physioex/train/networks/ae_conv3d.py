from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SleepAutoEncoderModule

module_config = dict()


class AutoEncoderConv3D(SleepAutoEncoderModule):
    def __init__(self, module_config=module_config):
        super(AutoEncoderConv3D, self).__init__(Net(module_config), module_config)

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.seq_len = module_config["seq_len"]
        self.in_channels = module_config["in_channels"]
        self.encoder = Encoder(module_config)
        self.decoder = Decoder(module_config)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        #print("after permute:", x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1, 3, 4)
        #print("after permute:", x.shape)
        return x

class Encoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.act_fn = nn.ReLU()
        in_c = config["in_channels"]
        seq_l = config["seq_len"]
        self.output_dim = config["latent_dim"]

        # self.lc_p1 = nn.Conv3d(1 * in_c, 1 * in_c, kernel_size = (3, 5, 11), stride=(1,1,1), padding=(1, 2, 5))
        # self.lc_p2 = nn.Conv3d(1 * in_c, 1 * in_c, kernel_size = (3, 5, 11), stride=(1,1,1), padding=(1, 2, 5))
        # self.lc_p3 = nn.Conv3d(1 * in_c, 1 * in_c, kernel_size = (3, 5, 11), stride=(1,1,1), padding=(1, 2, 5))
        # self.lc_p4 = nn.Conv3d(1 * in_c, 1 * in_c, kernel_size = (3, 5, 11), stride=(1,1,1), padding=(1, 2, 5))
        # self.lc_p5 = nn.Conv3d(1 * in_c, 1 * in_c, kernel_size = (3, 5, 11), stride=(1,1,1), padding=(1, 2, 5))

        self.lc1 = nn.Conv3d(1 * in_c, 2 * in_c, kernel_size = (2, 5, 11), stride=(1,1,2), padding=(0, 2, 5))
        self.lc2 = nn.Conv3d(2 * in_c, 4 * in_c, kernel_size = (2, 3, 3), stride=(1,2,2), padding=(0, 1, 1))
        self.lc3 = nn.Conv3d(4 * in_c, 8 * in_c, kernel_size = (1, 3, 3), stride=(1,2,2), padding=(0, 1, 1))
        self.lc4 = nn.Conv3d(8 * in_c, 8 * in_c, kernel_size = (1, 3, 3), stride=(1,2,2), padding=(0, 1, 1))
        self.ln = nn.Linear(288, self.output_dim)

    def forward(self, x):
        # x = self.lc_p1(x)
        # x = self.act_fn(x)
        # x = self.lc_p2(x)
        # x = self.act_fn(x)
        # x = self.lc_p3(x)
        # x = self.act_fn(x)
        # x = self.lc_p4(x)
        # x = self.act_fn(x)
        # x = self.lc_p5(x)
        # x = self.act_fn(x)

        x = self.lc1(x)
        #print("after lc1", x.shape)
        x = self.act_fn(x)
        x = self.lc2(x)
        #print("after lc2", x.shape)
        x = self.act_fn(x)
        x = self.lc3(x)
        #print("after lc3", x.shape)
        x = self.act_fn(x)
        x = self.lc4(x)
        #print("after lc4", x.shape)
        x = self.act_fn(x)
        x = x.view(x.size(0), -1)
        #print("after view", x.shape)
        x = self.ln(x)
        #print("after ln", x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.act_fn = nn.ReLU()
        in_c = config["in_channels"]
        self.input_dim = config["latent_dim"]

        self.lc1 = nn.ConvTranspose3d(8 * in_c, 8 * in_c, kernel_size = (1, 3, 3), stride=(1,2,2), padding=(0, 1, 1))
        self.lc2 = nn.ConvTranspose3d(8 * in_c, 4 * in_c, kernel_size = (1, 3, 3), stride=(1,2,2), padding=(0, 0, 1))
        self.lc3 = nn.ConvTranspose3d(4 * in_c, 2 * in_c, kernel_size = (2, 3, 3), stride=(1,2,2), padding=(0, 1, 1))
        self.lc4 = nn.ConvTranspose3d(2 * in_c, 1 * in_c, kernel_size = (2, 5, 11), stride=(1,1,2), padding=(0, 2, 5))
        self.ln = nn.Linear(self.input_dim, 288)

    def forward(self, x):
        x = self.ln(x)
        #print("after ln", x.shape)
        x = x.view(x.size(0), 8, 1, 4, 9)
        #print("after view", x.shape)
        x = self.lc1(x)
        #print("after lc1", x.shape)
        x = self.act_fn(x)
        x = self.lc2(x)
        #print("after lc2", x.shape)
        x = self.act_fn(x)
        x = self.lc3(x)
        #print("after lc3", x.shape)
        x = self.act_fn(x)
        x = self.lc4(x)
        #print("after lc4", x.shape)
        x = self.act_fn(x)

        return x

    