import os

import pkg_resources as pkg
import yaml

from physioex.train.networks.chambon2018 import Chambon2018Net
from physioex.train.networks.seqsleepnet import SeqSleepNet
from physioex.train.networks.tinysleepnet import TinySleepNet
from physioex.train.networks.seqsleepnet_cem import SeqSleepNetCEM
from physioex.train.networks.tinysleepnet import TinySleepNet
from physioex.train.networks.ae_fullyconnected import AutoEncoderFullyConnected
from physioex.train.networks.ae_conv3d import AutoEncoderConv3D
from physioex.train.networks.ae_seqsleepnet import AutoEncoderSeqSleepNet
from physioex.train.networks.vae_seqsleepnet import VAESeqSleepNet
from physioex.train.networks.protoae_seqsleepnet import PrototypeAESeqSleepNet
from physioex.train.networks.wrapper import Wrapper


config_file = pkg.resource_filename(
    "physioex", os.path.join("train", "networks", "config.yaml")
)
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Network configuration file not found: {config_file}")


with open(config_file, "r") as file:
    config = yaml.safe_load(file)
