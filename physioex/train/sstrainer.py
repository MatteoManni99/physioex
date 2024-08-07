from typing import List

import uuid
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from joblib import Parallel, delayed
from lightning.pytorch import seed_everything
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger

from physioex.data import PhysioExDataModule, PhysioExDataset, get_datasets
from physioex.train.networks import get_config
from physioex.train.networks.utils.loss import config as loss_config

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)

class SelfSupervisedTrainer:
    def __init__(
        self,
        datasets: List[str] = ["mass"],
        versions: List[str] = None,
        batch_size: int = 32,
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 1,
        data_folder: str = None,

        random_fold: bool = False,

        model_name: str = "ae_seqsleepnet",

        loss_name: str = "mse",

        ckp_path: str = None,
        max_epoch: int = 20,
        val_check_interval: int = 3,
        #n_jobs: int = 10,
        penalty_change: bool = False,
    ):
        ###### module setup ######
        network_config = get_config()[model_name]
        
        module_config = network_config["module_config"]
        module_config["seq_len"] = sequence_length
        module_config["loss_call"] = loss_config[loss_name]
        module_config["loss_params"] = dict()
        module_config["loss_name"] = loss_name
        module_config["in_channels"] = len(selected_channels)
        module_config["batch_size"] = batch_size

        self.model_call = network_config["module"]
        self.module_config = module_config
        
        ###### datamodule setup ######
        if random_fold :
            self.folds = [ -1 ]
        else:
            self.folds = PhysioExDataset(
                datasets=datasets,
                versions=versions,
                preprocessing=network_config["input_transform"],
                selected_channels=selected_channels,
                sequence_length=sequence_length,
                target_transform=network_config["target_transform"],
                data_folder=data_folder,
            ).get_num_folds()
            
            self.folds = list(range(self.folds))

        num_steps = PhysioExDataset(
            datasets=datasets,
            versions=versions,
            preprocessing=network_config["input_transform"],
            selected_channels=selected_channels,
            sequence_length=sequence_length,
            target_transform=network_config["target_transform"],
            data_folder=data_folder,
        ).__len__() // batch_size
        
        # print("Num steps: ", num_steps)
        # print("Val check interval: ", val_check_interval)
        # print("batch size: ", batch_size)
        # val_check_interval = max(1, num_steps // val_check_interval)

        self.datasets = datasets
        self.versions = versions
        self.batch_size = batch_size
        self.preprocessing = network_config["input_transform"]
        self.selected_channels = selected_channels
        self.sequence_length = sequence_length
        self.data_folder = data_folder
        self.target_transform = network_config["target_transform"]
        
        ##### trainer setup #####

        self.penalty_change = penalty_change
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.val_check_interval = val_check_interval

        self.ckp_path = ckp_path if ckp_path is not None else "models/" + str(uuid.uuid4()) + "/"
        Path(self.ckp_path).mkdir(parents=True, exist_ok=True)
        
        #############################



    def train_evaluate(self, fold: int = 0):

        logger.info(
            "JOB:%d-Splitting dataset into train, validation and test sets" % fold
        )
        
        #### datamodules setup ####

        # datamodule = TimeDistributedModule(
        #     dataset=dataset,
        #     batch_size=self.batch_size,
        #     fold=fold,
        # )

        train_datamodule = PhysioExDataModule(
            datasets=self.datasets,
            versions=self.versions,
            folds=fold,
            batch_size=self.batch_size,
            selected_channels=self.selected_channels,
            sequence_length=self.sequence_length,
            data_folder=self.data_folder,
            preprocessing = self.preprocessing,
            target_transform= self.target_transform,
        )
        
        ###### module setup ######

        module = self.model_call(module_config=self.module_config)

        ###### trainer setup ######

        # Definizione delle callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=self.ckp_path,
            filename="fold=%d-{epoch}-{step}-{val_loss:.2f}" % fold,
            save_weights_only=False,
        )
        
        progress_bar_callback = RichProgressBar()

        my_logger = CSVLogger(save_dir=self.ckp_path)

        # Configura le callback
        if self.penalty_change:
            callbacks_list = [checkpoint_callback, progress_bar_callback, ChangeLossCallback()]
        else:
            callbacks_list = [checkpoint_callback, progress_bar_callback]

        # Configura il trainer con le callback
        trainer = pl.Trainer(
            devices="auto",
            max_epochs=self.max_epoch,
            val_check_interval=self.val_check_interval,
            callbacks=callbacks_list,
            deterministic=True,
            logger=my_logger,
        )

        ###### training ######

        logger.info("JOB:%d-Training model" % fold)
        # Addestra il modello utilizzando il trainer e il DataModule
        trainer.fit(module, datamodule=train_datamodule)

        logger.info("JOB:%d-Evaluating model" % fold)
        val_results = trainer.test(
            ckpt_path="best", dataloaders=train_datamodule.val_dataloader()
        )[0]

        val_results["fold"] = fold

        test_results = trainer.test(ckpt_path="best", datamodule=train_datamodule)[0]

        return {"val_results": val_results, "test_results": test_results}

    def run(self):

        results = [self.train_evaluate(fold) for fold in self.folds]

        val_results = pd.DataFrame([result["val_results"] for result in results])
        test_results = pd.DataFrame(
            [result["test_results"] for result in results]
        )

        val_results.to_csv(self.ckp_path + "val_results.csv", index=False)
        test_results.to_csv(self.ckp_path + "test_results.csv", index=False)
        
        logger.info("Results successfully saved in %s" % self.ckp_path)


class ChangeLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.last_loss = float('inf')
    
    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics['val_loss']
        print("Val loss: ", val_loss)
        print("Last loss: ", self.last_loss)
        if(abs(val_loss - self.last_loss) < 0.02):
            print("Loss is not improving")
            pl_module.penalty = not(pl_module.penalty)
            print("Penalty is now: ", pl_module.penalty)
        self.last_loss = val_loss
