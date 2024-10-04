import os

import pkg_resources as pkg
import yaml

from physioex.train.networks.chambon2018 import Chambon2018Net
from physioex.train.networks.seqsleepnet import SeqSleepNet
#------------ROBA VECCHIA----------------
from physioex.train.networks.seqsleepnet_cem import SeqSleepNetCEM
from physioex.train.networks.sleeptransformer import SleepTransformer
from physioex.train.networks.tinysleepnet import TinySleepNet
from physioex.train.networks.ae_fullyconnected import AutoEncoderFullyConnected
from physioex.train.networks.ae_conv3d import AutoEncoderConv3D
from physioex.train.networks.ae_seqsleepnet import AutoEncoderSeqSleepNet
from physioex.train.networks.vae_seqsleepnet import VAESeqSleepNet
from physioex.train.networks.protoae_seqsleepnet import PrototypeAESeqSleepNet
from physioex.train.networks.wrapper import Wrapper
# from physioex.train.networks.seqecgnet import SeqECGnet


def read_config(model_name: str):
    config_file = pkg.resource_filename(__name__, "config/" + model_name + ".yaml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config


config = {
    "chambon2018": {
        "module_config": read_config("chambon2018"),
        "module": Chambon2018Net,
        "input_transform": "raw",
        "target_transform": target_transform.get_mid_label,
    },
    "seqsleepnet": {
        "module_config": read_config("seqsleepnet"),
        "module": SeqSleepNet,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "sleeptransformer": {
        "module_config": read_config("sleeptransformer"),
        "module": SleepTransformer,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "tinysleepnet": {
        "module_config": read_config("tinysleepnet"),
        "module": TinySleepNet,
        "input_transform": "raw",
        "target_transform": None,
    },
    "seqsleepnet_cem": {
        "module_config": read_config("seqsleepnet_cem"),
        "module": SeqSleepNetCEM,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "ae_fullyconnected": {
        "module_config": read_config("ae_fullyconnected"),
        "module": AutoEncoderFullyConnected,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "ae_conv3d": {
        "module_config": read_config("ae_conv3d"),
        "module": AutoEncoderConv3D,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "ae_seqsleepnet": {
        "module_config": read_config("ae_seqsleepnet"),
        "module": AutoEncoderSeqSleepNet,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "vae_seqsleepnet": {
        "module_config": read_config("vae_seqsleepnet"),
        "module": VAESeqSleepNet,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "protoae_seqsleepnet": {
        "module_config": read_config("protoae_seqsleepnet"),
        "module": PrototypeAESeqSleepNet,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "wrapper": {
        "module_config": read_config("wrapper"),
        "module": Wrapper,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
}


def get_config():
    return config


def register_experiment(experiment: str = None):
    global config

    logger.info(f"Registering experiment {experiment}")

    try:
        with open(experiment, "r") as f:
            experiment = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Experiment {experiment} not found")

    experiment = experiment["experiment"]

    experiment_name = experiment["name"]

    config[experiment_name] = dict()

    module = importlib.import_module(experiment["module"])

    config[experiment_name]["module"] = getattr(module, experiment["class"])
    config[experiment_name]["module_config"] = experiment["module_config"]
    config[experiment_name]["input_transform"] = experiment["input_transform"]

    if experiment["target_transform"] is not None:
        if experiment["module"] != experiment["target_transform"]["module"]:
            module = importlib.import_module(experiment["target_transform"]["module"])

        config[experiment_name]["target_transform"] = getattr(
            module, experiment["target_transform"]["function"]
        )
    else:
        logger.warning(f"Target transform not found for {experiment_name}")
        config[experiment_name]["target_transform"] = None

    return experiment_name
#------------------------FINE ROBA VECCHIA---------------------

from physioex.train.networks.tinysleepnet import TinySleepNet

config_file = pkg.resource_filename(
    "physioex", os.path.join("train", "networks", "config.yaml")
)
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Network configuration file not found: {config_file}")


with open(config_file, "r") as file:
    config = yaml.safe_load(file)
