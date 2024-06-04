from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from physioex.train.networks.base import SleepAutoEncoderModule
from physioex.train.networks.seqsleepnet import EpochEncoder, SequenceEncoder

module_config = dict()


class AutoEncoderSeqSleepNet(SleepAutoEncoderModule):
    def __init__(self, module_config=module_config):
        super(AutoEncoderSeqSleepNet, self).__init__(Net(module_config), module_config)

class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()
        self.L = module_config["seq_len"]
        self.nchan = module_config["in_channels"]
        self.T = module_config["T"]
        self.F = module_config["F"]

        self.epoch_encoder = EpochEncoder(module_config)
        self.sequence_encoder = SequenceEncoder(module_config)

        self.lin_encode = nn.Linear(2 * module_config["seqnhidden2"], module_config["latent_dim"])
        self.lin_decode = nn.Linear(module_config["latent_dim"], 2 * module_config["seqnhidden2"])

        self.layer_norm = nn.LayerNorm(module_config["latent_dim"])
        self.layer_norm2 = nn.LayerNorm(module_config["seqnhidden1"]*2)
        
        self.sequence_decoder = SequenceDecoder(module_config)
        self.epoch_decoder = EpochDecoder2(module_config)

    def encode(self, x):
        batch = x.size(0)
        #print("input:", x.size())
        x = x.view(-1, self.nchan, self.T, self.F)
        #print("reshape:", x.size())
        x = self.epoch_encoder(x)
        #print("after epoch encoder:", x.size())
        x = x.view(batch, self.L, -1)
        #print("reshape:", x.size())
        x = self.sequence_encoder.encode(x)
        #print("after sequence encoder:", x.size())
        x = self.lin_encode(x)
        #print("after lin encode:", x.size())
        return x
    
    def encode_norm(self, x):
        batch = x.size(0)
        #print("input:", x.size())
        x = x.view(-1, self.nchan, self.T, self.F)
        #print("reshape:", x.size())
        x = self.epoch_encoder(x)
        #print("after epoch encoder:", x.size())
        x = x.view(batch, self.L, -1)
        #print("reshape:", x.size())
        x = self.sequence_encoder.encode(x)
        #print("after sequence encoder:", x.size())
        x = self.lin_encode(x)
        x = self.layer_norm(x)
        #print("after lin encode:", x.size())
        return x
    
    def encode_norm2(self, x):
        batch = x.size(0)
        #print("input:", x.size())
        x = x.view(-1, self.nchan, self.T, self.F)
        #print("reshape:", x.size())
        x = self.epoch_encoder(x)
        #print("after epoch encoder:", x.size())
        x = x.view(batch, self.L, -1)
        
        x = self.layer_norm2(x)
        #print("reshape:", x.size())
        x = self.sequence_encoder.encode(x)
        #print("after sequence encoder:", x.size())
        x = self.lin_encode(x)
        x = self.layer_norm(x)
        #print("after lin encode:", x.size())
        return x

    def decode(self, x):
        batch = x.size(0) #x.shape = [b, l, latent_dim]
        x = self.lin_decode(x)
        #print("after lin decode:", x.size())
        x = self.sequence_decoder(x)
        #print("after sequence decoder:", x.shape)
        x = x.reshape(batch * self.L, -1) #x.shape = [b*l, 128]
        #print("reshape:", x.size())
        x = self.epoch_decoder(x)
        #print("after epoch decoder:", x.size())
        x = x.reshape(batch, self.L, self.nchan, self.T, self.F)
        return x

    def forward(self, x):
        x = self.encode_norm2(x)
        x = self.decode(x)
        return x

class SequenceDecoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.gru_seq_decoder = nn.LSTM(
            input_size=config["seqnhidden2"] * 2,
            hidden_size=config["seqnhidden1"],
            num_layers=config["seqnlayer2"],
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        x, _= self.gru_seq_decoder(x)
        return x


class EpochDecoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        #[96, 128]
        #[96, 1, 29, 129]
        self.nchan = config["in_channels"]
        self.D = config["D"]
        self.T = config["T"]
        self.F = config["F"]
        # self.gru_epoch_decoder = nn.LSTM(
        #     input_size=config["seqnhidden1"] * 2,
        #     hidden_size=config["D"] * self.nchan,
        #     num_layers=config["seqnlayer1"],
        #     batch_first=True,
        #     bidirectional=True,
        # )
        self.fc1 = nn.Linear(128, 16 * 4 * 16)
        #self.fc2 = nn.Linear(256, 256)
        #self.fc3 = nn.Linear(256, 16 * 4 * 16)  # Adattare per il successivo reshaping
        
        # Strati convoluzionali trasposti
        self.conv_transpose1 = nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1,0))
        self.conv_transpose4 = nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose5 = nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose2 = nn.ConvTranspose2d(8, 4, kernel_size=(3, 5), stride=(2, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose3 = nn.ConvTranspose2d(4, 1, kernel_size=(3, 5), stride=(2, 2), padding=(1, 0), output_padding=(0,0))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Livelli fully connected
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.relu(self.fc3(x))
        
        # Reshape per adattarsi al primo strato conv2d trasposto
        x = x.view(-1, 16, 4, 16)

        # Strati convoluzionali trasposti
        x = self.relu(self.conv_transpose1(x))
        #print("t1", x.size())
        x = self.relu(self.conv_transpose4(x))
        #print("t4", x.size())
        x = self.relu(self.conv_transpose5(x))
        #print("t5", x.size())
        x = self.relu(self.conv_transpose2(x))
        #print("t2", x.size())
        x = self.conv_transpose3(x)
        #print("t3", x.size())
        
        return x

class EpochDecoder2(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        #[96, 128]
        #[96, 1, 29, 129]
        self.nchan = config["in_channels"]
        self.D = config["D"]
        self.T = config["T"]
        self.F = config["F"]
        self.gru_epoch_decoder = nn.GRU(
            input_size=16,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(128, 16 * 8 * 16)
        #self.fc2 = nn.Linear(1*29*64, 1*29*129)
        #self.fc3 = nn.Linear(256, 16 * 4 * 16)  # Adattare per il successivo reshaping
        
        # Strati convoluzionali trasposti
        self.conv_transpose1 = nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose12 = nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose13 = nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose2 = nn.ConvTranspose2d(8, 4, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1), output_padding=(0,0))
        self.conv_transpose3 = nn.ConvTranspose2d(4, 1, kernel_size=(3, 5), stride=(2, 2), padding=(1, 0), output_padding=(0,0))
        self.relu = nn.ReLU()

    def forward(self, x):
        #print(x.size())
        # Livelli fully connected
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.relu(self.fc3(x))
        #print(x.size())
        x = x.view(-1, 8, 16)
        #print(x.size())
        x, _ = self.gru_epoch_decoder(x) # [batch*L, T, F*nchan]
        #print(x.size())
        forward_output = x[:, :, :16]
        backward_output = x[:, :, 16:]
        x = forward_output + backward_output

        #print(x.size())
        # Reshape per adattarsi al primo strato conv2d trasposto
        x = x.view(-1, 16, 8, 16)
        #print(x.size())
        # Strati convoluzionali trasposti
        x = self.relu(self.conv_transpose1(x))
        #print("t1", x.size())
        # x = self.relu(self.conv_transpose12(x))
        # #print("t2", x.size())
        # x = self.relu(self.conv_transpose13(x))
        #print("t1", x.size())
        x = self.relu(self.conv_transpose2(x))
        #print("t2", x.size())
        x = self.conv_transpose3(x)
        #print("t3", x.size())
        
        return x



# class EpochDecoder(nn.Module):
#     def __init__(self, config: Dict):
#         super().__init__()
#         [96, 128]
#         [96, 1, 29, 129]
#         self.nchan = config["in_channels"]
#         self.D = config["D"]
#         self.T = config["T"]
#         self.F = config["F"]
#         self.gru_epoch_decoder = nn.LSTM(
#             input_size=config["seqnhidden1"] * 2,
#             hidden_size=config["D"] * self.nchan,
#             num_layers=config["seqnlayer1"],
#             batch_first=True,
#             bidirectional=True,
#         )
#         self.transposeConvNet = TransposeConvNet(self.nchan, T=config["T"], D=config["D"]*2, F=config["F"])
#         self.layer_norm = nn.LayerNorm([self.nchan, self.T, self.F])

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = x.repeat(1, 14, 1)
#         print(x.size())  # torch.Size([96, 14, 128])
#         x, _ = self.gru_epoch_decoder(x) # [batch*L, T, D*nchan]
#         x = x.reshape(-1, 14, self.nchan, self.D * 2)
#         x = x.permute(0, 2, 1, 3) # [batch*L, nchan, T, D*2]
#         print(x.size())
#         x = self.transposeConvNet(x)
#         x = self.layer_norm(x)
#         return x

# class TransposeConvNet(nn.Module):
#     def __init__(self, nchan, T, D, F):
#         super(TransposeConvNet, self).__init__()
#         self.nchan = nchan
#         self.T = T
#         self.D = D
#         self.F = F

#         First ConvTranspose2d layer parameters
#         self.conv_transpose1 = nn.ConvTranspose2d(
#             in_channels=self.nchan,
#             out_channels=self.nchan*16,
#             kernel_size=(5, 5),
#             stride=(2, 2),
#             padding=(1, 1),
#         )
#         self.conv_transpose2 = nn.ConvTranspose2d(
#             in_channels=self.nchan*16,
#             out_channels=self.nchan*16,
#             kernel_size=(3, 3),
#             stride=(1, 1),
#             padding=(1, 1),
#         )
#         self.conv_transpose3 = nn.ConvTranspose2d(
#             in_channels=self.nchan*16,
#             out_channels=self.nchan,
#             kernel_size=(3, 3),
#             stride=(1, 1),
#             padding=(1, 1),
#         )


#     def forward(self, x):
#         print("input",x.size())
#         x = self.conv_transpose1(x)
#         x = nn.LeakyReLU()(x)
#         print("conv1:", x.size())
#         x = self.conv_transpose2(x)
#         x = nn.LeakyReLU()(x)
#         print("conv2:", x.size())
#         x = self.conv_transpose3(x)
#         print("conv3:", x.size())
#         return x
    