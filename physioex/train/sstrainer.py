import copy
import uuid
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from joblib import Parallel, delayed
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks import Callback

from physioex.data import TimeDistributedModule, get_datasets
from physioex.train.networks import get_config
from physioex.train.networks.utils.loss import config as loss_config


import torch
from loguru import logger

torch.set_float32_matmul_precision("medium")


class SelfSupervisedTrainer:
    def __init__(
        self,
        model_name: str = "ae_fullyconnected",
        dataset_name: str = "sleep_physioex",
        version: str = "2018",
        sequence_length: int = 1,
        picks: list = ["Fpz-Cz"],
        loss_name: str = "mse",
        ckp_path: str = None,
        max_epoch: int = 20,
        val_check_interval: int = 300,
        batch_size: int = 32,
        n_jobs: int = 10,
        penalty_change: bool = False,
        #imbalance: bool = False,
    ):

        seed_everything(42, workers=True)

        datasets = get_datasets()
        config = get_config()

        self.dataset_call = datasets[dataset_name]
        self.model_call = config[model_name]["module"]
        self.input_transform = config[model_name]["input_transform"]
        #self.target_transform = config[model_name]["target_transform"]
        self.module_config = config[model_name]["module_config"]
        self.module_config["seq_len"] = sequence_length
        self.module_config["batch_size"] = batch_size

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.val_check_interval = val_check_interval
        self.version = version
        self.n_jobs = n_jobs
        self.penalty_change = penalty_change
        self.loss_name = loss_name

        if ckp_path is None:
            self.ckp_path = "models/" + str(uuid.uuid4()) + "/"
        else:
            self.ckp_path = ckp_path

        Path(self.ckp_path).mkdir(parents=True, exist_ok=True)

        picks = picks.split(" ")
        self.module_config["picks"] = picks
        self.module_config["in_channels"] = len(picks)
        
        logger.info("Loading dataset")
        self.dataset = self.dataset_call(
            version=self.version,
            picks=picks,
            preprocessing=self.input_transform,
            sequence_length=sequence_length,
            target_transform=None,
        )

        logger.info("Dataset loaded")

        self.folds = list(range(self.dataset.get_num_folds()))

        self.module_config["loss_call"] = loss_config[loss_name]
        self.module_config["loss_name"] = loss_name

        #self.module_config["loss_params"] = dict()

    def train_evaluate(self, fold: int = 0):

        dataset = self.dataset

        logger.info(
            "JOB:%d-Splitting dataset into train, validation and test sets" % fold
        )
        dataset.split(fold)

        datamodule = TimeDistributedModule(
            dataset=dataset,
            batch_size=self.batch_size,
            fold=fold,
        )

        module = self.model_call(module_config=self.module_config)

        # Definizione delle callback
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=self.ckp_path,
            filename="fold=%d-{epoch}-{step}-{val_loss:.2f}" % fold,
        )
        
        progress_bar_callback = RichProgressBar()

        if self.penalty_change:
            callbacks_list = [checkpoint_callback, progress_bar_callback, ChangeLossCallback()]
        else:
            callbacks_list = [checkpoint_callback, progress_bar_callback]

        # Configura il trainer con le callback
        trainer = pl.Trainer(
            max_epochs=self.max_epoch,
            val_check_interval=self.val_check_interval,
            callbacks=callbacks_list,
            deterministic=True,
        )

        logger.info("JOB:%d-Training model" % fold)
        # Addestra il modello utilizzando il trainer e il DataModule
        trainer.fit(module, datamodule=datamodule)

        logger.info("JOB:%d-Evaluating model" % fold)
        val_results = trainer.test(
            ckpt_path="best", dataloaders=datamodule.val_dataloader()
        )
        test_results = trainer.test(ckpt_path="best", datamodule=datamodule)

        return {"val_results": val_results, "test_results": test_results}

    def run(self):
        logger.info("Jobs pool spawning")

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_evaluate)(fold) for fold in self.folds
        )

        # results = []
        # for fold in folds:
        #     results.append(self.train_evaluate(fold))
        #     gc.collect()

        logger.info("Results successfully collected from jobs pool")

        try:
            val_results = pd.DataFrame([result["val_results"][0] for result in results])
            test_results = pd.DataFrame(
                [result["test_results"][0] for result in results]
            )

            val_results.to_csv(self.ckp_path + "val_results.csv", index=False)
            test_results.to_csv(self.ckp_path + "test_results.csv", index=False)
        except Exception as e:
            logger.error(f"Error while saving results: {e}")
            raise e

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
