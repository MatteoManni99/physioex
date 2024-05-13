import pkg_resources as pkg
import yaml

import physioex as physioex
from physioex.train.ssnetworks.base_ae import BaseAutoEncoder


def read_config(model_name: str):
    config_file = pkg.resource_filename(__name__, "config/" + model_name + ".yaml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config

config = {
    "base_ae": {
        "module_config": read_config("base_ae"),
        "module": BaseAutoEncoder,
        "input_transform": "xsleepnet",
        #"target_transform": None,
    },
}
