import os
import shutil
from pathlib import Path
from typing import Callable, List

import numpy as np
from loguru import logger
from tqdm import tqdm
from torch import torch

from physioex.train.models import load_model

class ConceptLabeler:
    def __init__(
        self,
        dataset_name: str,
        data_folder: str,
        model_class: Callable,
        model_config: dict,
        model_ckpt_path: str,
        channels_index: List[int],
        sequence_length: int,
        lambda_fun = None,
    ):
        # change the current working directory to the root directory, this could not work in a different environment
        os.chdir(os.path.abspath(os.sep))

        if lambda_fun is None:
            self.lambda_fun = lambda d: 1/(10000**d)
        else:
            self.lambda_fun = lambda_fun

        self.model = load_model(
            model = model_class,
            model_kwargs = model_config,
            ckpt_path = model_ckpt_path,
        ).eval()
        
        self.data_folder = data_folder
        self.dataset_folder = os.path.join(self.data_folder, dataset_name)
        
        self.data_path = os.path.join(self.dataset_folder, "xsleepnet/")

        self.channels_index = channels_index
        self.L = sequence_length
        self.central_window = int((self.L - 1)/2)
        scaling = np.load(os.path.join(self.data_path, "scaling.npz"))
        
        self.mean = scaling["mean"]
        self.std = scaling["std"]
        self.input_shape = list(self.mean.shape)
        
        #self.concepts_path = os.path.join(self.dataset_folder, "concepts/")
        #self.distances_path = os.path.join(self.dataset_folder, "distances/")
        self.concepts_path = os.path.join("home/manni/", "concepts/")
        self.distances_path = os.path.join("home/manni/", "distances/")

        Path(self.concepts_path).mkdir(parents=True, exist_ok=True)
        Path(self.distances_path).mkdir(parents=True, exist_ok=True)

        self.max_dist_value = float("-inf")
        self.min_dist_value = float("inf")

    # run method to perform the concept labeling process on the dataset
    # this method is specifically designed to work with protoae_seqsleepnet model

    def run(self):
        logger.info("Running the concept labeler ...")
        
        # lambda function that computes the concept values from the distances
        

        prototypes = self.model.nn.classifier.prototypes.detach()

        num_windows_dict = {}

        logger.info("Computing the distances ...")
        for file in tqdm(os.listdir(self.data_path)):
            if file.endswith('.npy'):
                file_name, _ = os.path.splitext(file)
                subject_id = int(file_name)
                data_path = os.path.join(self.data_path, file)

                X = np.memmap(data_path, dtype="float32", mode="r")
                num_windows = X.shape[0]/np.prod(self.input_shape)
                num_windows_dict[subject_id] = int(num_windows)
                assert num_windows.is_integer(), "ERR: reading the data failed"
            
                X = (X.reshape(tuple([int(num_windows)] + self.input_shape)))[:, self.channels_index]
                X = (X - self.mean[self.channels_index]) / self.std[self.channels_index]

                # fill the first and last windows with the first and last window
                num_sequences = X.shape[0] - self.L + 1
                X_sliding = np.stack([X[i:i + self.L] for i in range(num_sequences)], axis=0)
                X_padded_start = np.repeat(X_sliding[0:1], repeats=((self.L - 1)/2), axis=0)
                X_padded_end = np.repeat(X_sliding[-1:], repeats=((self.L - 1)/2), axis=0)
                X_to_label = torch.tensor(np.concatenate([X_padded_start, X_sliding, X_padded_end], axis=0)).to(self.model.device)
                
                # compute the distances between the embeddings and the prototypes
                with torch.no_grad():
                    z = self.model.nn.encode(X_to_label)
                    z = z[:, self.central_window, :] # take the central window
                    
                    distances = torch.cdist(z, prototypes, p=2)
                    
                    if (distances.max().cpu().numpy() > self.max_dist_value):
                        self.max_dist_value = distances.max().cpu().numpy()
                    if (distances.min().cpu().numpy() < self.min_dist_value):
                        self.min_dist_value = distances.min().cpu().numpy()
                    
                    distances_path = os.path.join(self.distances_path, file)
                    distances_memmap = np.memmap(distances_path, dtype=np.float32, mode="w+", shape=distances.shape)
                    distances_memmap[:] = distances[:].cpu().numpy()
                    distances_memmap.flush()
            

        logger.info("Computing the concepts ...")
        for file in tqdm(os.listdir(self.distances_path)):
            file_name, _ = os.path.splitext(file)
            subject_id = int(file_name)
            distances_path = os.path.join(self.distances_path, file)       
            distances = np.memmap(distances_path, dtype=np.float32, mode="r", shape=(num_windows_dict[subject_id], prototypes.shape[0]))
            distances = (distances - self.min_dist_value)/(self.max_dist_value - self.min_dist_value)
            concepts = self.lambda_fun(distances)
            concepts_memmap = np.memmap(os.path.join(self.concepts_path, file), dtype=np.float32, mode="w+", shape=concepts.shape)
            concepts_memmap[:] = concepts[:]
            concepts_memmap.flush()

        np.savez(os.path.join(self.concepts_path, "concepts_shape"), shape=concepts.shape[1]) # save the number of concepts

        # clean up the distances folder after the concept labeling process
        shutil.rmtree(self.distances_path)
