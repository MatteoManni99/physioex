import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

import boto3
import h5py
import numpy as np
import pandas as pd
import pkg_resources as pkg
import tqdm
import yaml
from botocore import UNSIGNED
from botocore.client import Config
from dirhash import dirhash
from loguru import logger
from scipy.signal import butter, lfilter, resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from physioex.data.base import PhysioExDataset
from physioex.data.utils import read_cache, write_cache
from physioex.data.sleep_physionet import SleepPhysionet
from physioex.data.mitdb import MITBIH
from physioex.data.dreem import Dreem

datasets = {"sleep_physionet": SleepPhysionet, "dreem": Dreem, "mitdb": MITBIH}
n_subjects = {"sleep_physionet_2018": 83, "dreem_dodh": 25}

class Merged(PhysioExDataset):
    def __init__(
        self,
        version: str = "_",
        use_cache: bool = True,
        # preprocessors=[
        #     lambda data: np.multiply(data, 1e6),  # Convert from V to uV
        #     lambda data: butter_bandpass_filter(data, 0.3, 30, 250),
        #     lambda data: resample(data, 100 * 30),
        # ],
        #picks=["C3_M2"],
    ):
        self.config = read_config()
        
        # cache_path = "temp/merged"

        # for dataset_name in self.config["datasets_to_merge"]:
        #     cache_path += "_" + dataset_name
        #     if self.config["versions"][dataset_name]:
        #         cache_path += "_" + self.config["versions"][dataset_name]
        
        # cache_path += ".pkl"
        self.cache_path = "temp/merged/"
        print(self.cache_path)

        self.splits = {}
        self.set_random_splits()
        print(self.splits)

        self.window_dataset = None
        #self.version = version

        Path("temp/merged/").mkdir(parents=True, exist_ok=True)

        files_cached = True
        if use_cache:
            for dataset_name in self.config["datasets_to_merge"]:
                for fold in range(10):
                    cache_path = self.cache_path + dataset_name + "_" + self.config["versions"][dataset_name] + "_" + str(fold) + ".pkl"
                    if not os.path.exists(cache_path):
                        files_cached = False
                        break
            
        if files_cached:
            return
        
        logger.info("Fetching the dataset..")
        self.windows_dataset = np.empty((0, 2))
        self.datasets_to_merge = {}
        for dataset_name in self.config["datasets_to_merge"]:
            dataset = datasets[dataset_name](use_cache=use_cache, version=self.config["versions"][dataset_name])
            for fold in range(10):
                dataset.split(fold, self.splits[dataset_name])
                train, valid, test = dataset.get_sets()

                #TODO cercare un modo più elegante per togliere l'ultima colonna
                if (dataset_name == "sleep_physionet"):
                    temp = []
                    for i in range(len(train)):
                        temp.append((train[i][0], train[i][1]))
                    train = temp
                    temp = []
                    for i in range(len(valid)):
                        temp.append((valid[i][0], valid[i][1]))
                    valid = temp
                    temp = []
                    for i in range(len(test)):
                        temp.append((test[i][0], test[i][1]))
                    test = temp
                        
                cache_path = self.cache_path + dataset_name + "_" + self.config["versions"][dataset_name] + "_" + str(fold) + ".pkl"
                write_cache(cache_path, (train, valid, test))
                
                # if (dataset_name == "sleep_physionet"):
                #     #TODO cercare un modo più elegante per togliere l'ultima colonna
                #     dataset = []
                #     for i in range(len(temp_windows_dataset)):
                #         dataset.append((temp_windows_dataset[i][0], temp_windows_dataset[i][1]))
                #     temp_windows_dataset = dataset

                # if (dataset_name == "dreem"):
                #     temp_windows_dataset = [temp_windows_dataset["X"], temp_windows_dataset["y"]]
                #     X = np.concatenate(temp_windows_dataset[0], axis=0)
                #     y = np.concatenate(temp_windows_dataset[1], axis=0)
                #     temp_windows_dataset = []
                #     for i in range(len(y)):
                #         temp_windows_dataset.append((X[i], y[i]))
            
            # temp_windows_dataset = np.array(temp_windows_dataset, dtype=object)
            # self.windows_dataset = np.concatenate((self.windows_dataset, temp_windows_dataset), axis=0)

        #write_cache(cache_path, self.windows_dataset)


    def set_random_splits(self):
        for dataset in self.config["datasets_to_merge"]:
            n_sub = n_subjects[dataset + "_" + self.config["versions"][dataset]]
            s = np.arange(n_sub)
            np.random.shuffle(s)
            print(s)
            test_size = int(n_sub/10)
            print(test_size)
            train_subjects = []
            val_subjects = []
            test_subjects = []
            for i in range(10):
                test_set = s[test_size*i : test_size*i + test_size]
                
                valid_train_set = np.concatenate((s[0: test_size*i], s[test_size*i + test_size: n_sub]))
                if(i == 9):
                    val_set = valid_train_set[0: test_size]
                    train_set = valid_train_set[test_size: len(valid_train_set)]
                else:
                    val_set = valid_train_set[test_size*i: test_size*i + test_size]
                    train_set = np.concatenate((valid_train_set[0: test_size*i], valid_train_set[test_size*i + test_size: n_sub]))
                
                train_subjects.append(train_set)
                val_subjects.append(val_set)
                test_subjects.append(test_set)

            self.splits[dataset] = {"train": train_subjects, "valid": val_subjects, "test": test_subjects}


    def split(self, fold: int = 0):
        self.train_set = np.empty((0, 2))
        self.valid_set = np.empty((0, 2))
        self.test_set = np.empty((0, 2))

        for dataset_name in self.config["datasets_to_merge"]:
            cache_path = self.cache_path + dataset_name + "_" + self.config["versions"][dataset_name] + "_" + str(fold) + ".pkl"
            train_set_, valid_set_, test_set_ = read_cache(cache_path)
            
            self.train_set = np.concatenate((self.train_set, np.array(train_set_, dtype=object)), axis=0)
            self.valid_set = np.concatenate((self.valid_set, np.array(valid_set_, dtype=object)), axis=0)
            self.test_set = np.concatenate((self.test_set, np.array(test_set_, dtype=object)), axis=0)
        
        print("step1")
        
        #self.train_set, self.valid_set, self.test_set = train_set, valid_set, test_set


    def get_sets(self):
        return self.train_set, self.valid_set, self.test_set
    

@logger.catch
def read_config():
    config_file = pkg.resource_filename(__name__, "config/merged.yaml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":
    Merged()