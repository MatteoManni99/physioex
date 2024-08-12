import argparse
import importlib
from pathlib import Path

import yaml
from loguru import logger

from physioex.train import SelfSupervisedTrainer
from physioex.train.networks import config, register_experiment

def main():
    parser = argparse.ArgumentParser(description="Training script")

    # experiment arguments
    parser.add_argument(
        "-e",
        "--experiment",
        default="chambon2018",
        type=str,
        help='Specify the experiment to run. Expected type: str. Default: "chambon2018"',
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        default=None,
        type=str,
        help="Specify where to save the checkpoint. Expected type: str. Default: None",
    )

    parser.add_argument(
        "-cr",
        "--checkpoint_resume",
        default=None,
        type=str,
        help="Specify the checkpoint to resume training. Expected type: str. Default: None",
    )

    parser.add_argument(
        "-l",
        "--loss",
        default="mse",
        type=str,
        help='Specify the loss function to use. Expected type: str. Default: "cel" (Cross Entropy Loss)',
    )

    # dataset args
    parser.add_argument(
        "-d",
        "--dataset",
        default="mass",
        type=str,
        help='Specify the dataset to use. Expected type: str. Default: "SleepPhysionet"',
    )
    parser.add_argument(
        "-v",
        "--version",
        default=None,
        type=str,
        help='Specify the version of the dataset. Expected type: str. Default: "2018"',
    )
    parser.add_argument(
        "-p",
        "--picks",
        default="EEG",
        type=str,
        help="Specify the signal electrodes to pick to train the model. Expected type: list. Default: 'Fpz-Cz'",
    )

    # sequence
    parser.add_argument(
        "-sl",
        "--sequence_length",
        default=1,
        type=int,
        help="Specify the sequence length for the model. Expected type: int. Default: 3",
    )

    # trainer
    parser.add_argument(
        "-me",
        "--max_epoch",
        default=20,
        type=int,
        help="Specify the maximum number of epochs for training. Expected type: int. Default: 20",
    )
    parser.add_argument(
        "-vci",
        "--val_check_interval",
        default=300,
        type=int,
        help="Specify the validation check interval during training. Expected type: int. Default: 300",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="Specify the batch size for training. Expected type: int. Default: 32",
    )

    # parser.add_argument(
    #     "-nj",
    #     "--n_jobs",
    #     default=10,
    #     type=int,
    #     help="Specify the number of jobs for parallelization. Expected type: int. Default: 10",
    # )
    
    parser.add_argument(
        "--data_folder",
        "-df",
        type=str,
        default=None,
        required=False,
        help="The absolute path of the directory where the physioex dataset are stored, if None the home directory is used. Expected type: str. Optional. Default: None",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        help="Path to the configuration file in YAML format",
    )

    parser.add_argument(
        "--random_fold",
        "-rf",
        type=bool,
        default=False,
        required=False,
        help="Weather or not to perform the training on a random fold. Expected type: bool. Optional. Default: False",
    )

    parser.add_argument(
        "-pc",
        "--penalty_change",
        default=False,
        type=bool,
        help="Specify if the loss will change during training, adding or removing the penalty. Expected type: bool. Default: False",
    )

    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            # Override command line arguments with config file values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                    
    # check if the experiment is a yaml file
    if args.experiment.endswith(".yaml") or args.experiment.endswith(".yml"):
        args.experiment = register_experiment(args.experiment)

    # check if the dataset is a yaml file

    # convert the datasets into a list by diving by " "
    datasets = args.dataset.split(" ")
    versions = args.version.split(" ") if args.version is not None else None
    picks = args.picks.split(" ")
    
    print(datasets)
    print(versions)

    SelfSupervisedTrainer(
        model_name=args.experiment,
        datasets=datasets,
        versions = versions,
        ckp_path=args.checkpoint,
        data_folder=args.data_folder,
        loss_name=args.loss,
        selected_channels= picks,
        sequence_length=args.sequence_length,
        max_epoch=args.max_epoch,
        val_check_interval=args.val_check_interval,
        batch_size=args.batch_size,
        #n_jobs=args.n_jobs,
        random_fold = args.random_fold,
        penalty_change=args.penalty_change,
        resume_from_checkpoint = args.checkpoint_resume
    ).run()


if __name__ == "__main__":
    main()
