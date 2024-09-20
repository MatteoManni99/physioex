import os
import pickle
from typing import Callable, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from physioex.utils import get_data_folder
from physioex.data.datareader import DataReader
from loguru import logger

CHANNEL_INDEX = ["EEG", "EOG", "EMG", "ECG"]

DATASETS = {
    "mass": ["EEG", "EOG", "EMG"],
    "hmc": ["EEG", "EOG", "EMG", "ECG"],
    "shhs": ["EEG", "EOG", "EMG"],
    "dcsm": ["EEG", "EOG", "EMG", "ECG"],
    "mesa": ["EEG", "EOG", "EMG"],  #
    "mros": ["EEG", "EOG", "EMG"],  #
    "sleepedf": ["EEG", "EOG", "EMG"],  #    
}


class PhysioExDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List[str],
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        data_folder: str = None,
        hpc : bool = False,
    ):

        if data_folder is None:
            data_folder = get_data_folder()

        self.datasets = datasets
        self.L = sequence_length
        self.channels_index = [ CHANNEL_INDEX.index( ch ) for ch in selected_channels ]
        
        self.readers = []
        self.tables = []
        self.dataset_idx = []
        
        offset = 0
        for i, dataset in enumerate(datasets):
            reader = DataReader(
                data_folder = data_folder,
                dataset = dataset,
                preprocessing = preprocessing,
                sequence_length = sequence_length,
                channels_index = self.channels_index,
                offset = offset,
                hpc = hpc,
            )
            offset += len(reader)

            self.dataset_idx += list( np.ones( len(reader) ) * i ) 

            self.tables += [reader.get_table()]
            self.readers += [reader]
            
        self.dataset_idx = np.array( self.dataset_idx, dtype=np.uint8 )
        # set the table fold to the 0 fold by default
        self.split()
        self.target_transform = target_transform

        self.len = offset

    def __len__(self):
        return self.len

    def split(self, fold: int = -1, dataset_idx: int = -1):
        assert dataset_idx < len(self.tables), "ERR: dataset_idx out of range"

        # if fold is -1, set the split to a random fold for each dataset
        if fold == -1 and dataset_idx == -1:
            for i, table in enumerate(self.tables):
                num_folds = [col for col in table.columns if "fold_" in col]
                num_folds = len(num_folds)
                selcted_fold = np.random.randint(0, num_folds)

                self.tables[i]["split"] = table[f"fold_{selcted_fold}"].map(
                    {"train": 0, "valid": 1, "test": 2}
                )
        elif fold == -1 and dataset_idx != -1:
            num_folds = [
                col for col in self.tables[dataset_idx].columns if "fold_" in col
            ]
            num_folds = len(num_folds)
            selcted_fold = np.random.randint(0, num_folds)

            self.tables[dataset_idx]["split"] = table[f"fold_{selcted_fold}"].map(
                {"train": 0, "valid": 1, "test": 2}
            )
        elif fold != -1 and dataset_idx == -1:
            for i, table in enumerate(self.tables):
                self.tables[i]["split"] = table[f"fold_{fold}"].map(
                    {"train": 0, "valid": 1, "test": 2}
                )
        else:
            self.tables[dataset_idx]["split"] = table[f"fold_{fold}"].map(
                {"train": 0, "valid": 1, "test": 2}
            )

    def get_num_folds(self):
        # take the min number of folds for each dataset table
        num_folds = 100
        for table in self.tables:
            num_folds = min(
                num_folds, len([col for col in table.columns if "fold_" in col])
            )
        return num_folds

    def __getitem__(self, idx):

        dataset_idx = int(self.dataset_idx[idx])
        
        X, y = self.readers[dataset_idx][idx]
        
        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y

    def get_sets(self):
        # return the indexes in the table of the train, valid and test subjects
        train_idx = []
        valid_idx = []
        test_idx = []

        start_index = 0

        for table in self.tables:
            for _, row in table.iterrows():

                num_windows = row["num_windows"] - self.L + 1

                indices = np.arange(
                    start=start_index, stop=start_index + num_windows
                ).astype(np.uint32)

                start_index += num_windows

                if row["split"] == 0:
                    train_idx.append(indices)
                elif row["split"] == 1:
                    valid_idx.append(indices)
                elif row["split"] == 2:
                    test_idx.append(indices)
                else:
                    raise ValueError("ERR: split should be 0, 1 or 2")

        train_idx = np.concatenate(train_idx)
        valid_idx = np.concatenate(valid_idx)
        test_idx = np.concatenate(test_idx)

        return train_idx, valid_idx, test_idx
