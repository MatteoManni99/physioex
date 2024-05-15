import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Callable, List

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
from physioex.data.sleep_edf.sleep_edf import SleepEDF
from physioex.data.dreem.dreem import Dreem
from physioex.data.shhs.shhs import Shhs
from physioex.data.utils import read_config

dataset_class = {"sleep_physionet": SleepEDF, "dreem": Dreem, "shhs": Shhs}

equivalent_picks = {"dreem_EEG": "C3-M2", "dreem_EOG": "EOG", "dreem_EMG": "EMG"}
equivalent_picks.update({"sleep_physionet_EEG": "Fpz-Cz", "sleep_physionet_EOG": "EOG", "sleep_physionet_EMG": "EMG"})
equivalent_picks.update({"shhs_EEG": "EEG", "shhs_EOG": "EOG", "shhs_EMG": "EMG"})

datasets_to_merge = ["sleep_physionet", "dreem"]
versions_to_merge = {"dreem": "dodh", "sleep_physionet": "2018"}

class SleepMerged(PhysioExDataset):
    def __init__(
        self,
        version: str = None,
        picks: List[str] = ["EEG"],  # available [ "EEG", "EOG", "EMG" ]
        preprocessing: str = "raw",  # available [ "raw", "xsleepnet" ]
        sequence_length: int = 1,
        target_transform: Callable = None,
    ):
        #self.config = read_config("config/merged.yaml")

        self.datasets = []
        self.offsets = []
        #TODO implement the possibility to use more then one pick
        for dataset_name in datasets_to_merge:
            self.datasets.append(
                dataset_class[dataset_name](
                    version=versions_to_merge[dataset_name],
                    picks=[equivalent_picks[dataset_name + "_" + picks[0]]],
                    preprocessing=preprocessing,
                    sequence_length=sequence_length,
                    target_transform=target_transform,
                )
            )
        
        self.offsets.append(0)
        for dataset in self.datasets:
            self.offsets.append(self.offsets[-1] + len(dataset))
    
    
    def get_num_folds(self):
        #for dataset in self.datasets:
        #    assert dataset.get_num_folds() == 10
        #return 10
        return 1
    
    def split(self, fold: int = 0):
        for dataset in self.datasets:
            dataset.split(fold)
    
    def __getitem__(self, idx):
        i = 0
        for dataset in self.datasets:
            if self.offsets[i] <= idx and idx < self.offsets[i+1]:
                return dataset[idx - self.offsets[i]]
            i+=1
        raise IndexError("Wrong index")
    
    #redifine PhysioExDataset methods
    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
    
    def get_sets(self):
        total_train_idx = []
        total_valid_idx = []
        total_test_idx = []

        i = 0
        for dataset in self.datasets:
            train_idx, valid_idx, test_idx = dataset.get_sets()
            total_train_idx.append(train_idx + self.offsets[i])
            total_valid_idx.append(valid_idx + self.offsets[i])
            total_test_idx.append(test_idx + self.offsets[i])
            i+=1

        return np.concatenate(total_train_idx), np.concatenate(total_valid_idx), np.concatenate(total_test_idx)