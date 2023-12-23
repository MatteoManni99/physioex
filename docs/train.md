# Train state-of-the-art Pytorch Models

PhysioEx provides a fast and customizable way to train, evaluate and save state-of-the-art models for different physiological signal analysis tasks with different physiological signal datasets. This functionality is provided by the `train` command provided by this repository.

## Setup

Before using the `train` command, you need to set up a virtual environment and install the package in development mode. Here are the steps:

1. Make sure to have anaconda or miniconda correctly installed in your machine, then start installing a new virtual enviroment
```bash
    conda create -n myenv python==3.10
```    

2. Now jump into the enviroment and upgrade pip
```bash
    conda activate myenv
    conda install pip
    pip install --upgrade pip
```

3. Last but not least install PhysioEx in development mode
```bash
    git clone https://github.com/guidogagl/physioex.git
    cd physioex
    pip install -e .
```    

## Experiments
Currently, there are two experiments available in the repository:

- `chambon2018`: This experiment uses the [Chambon2018](https://ieeexplore.ieee.org/document/8307462) model for sleep stage classification.
- `tinysleepnet`: This experiment uses the [TinySleepNet](https://github.com/akaraspt/tinysleepnet)
 model for sleep stage classification.

To run an experiment, use the `-e` or `--experiment` argument followed by the name of the experiment. For example:

```bash
train --experiment chambon2018
```
### Dataset-experiment compatibility
|               | SleepPhysioNet | Dreem |
|---------------|:---------:|:-----:|
| chambon2018   |     ✔️     |   ✔️   |
| tinysleepnet  |     ✔️     |   ✔️   |

## Train Command
The train command is used to train models. Here are the available arguments:

- `-e`, `--experiment`: Specify the experiment to run. Expected `type: str`. `Default: "chambon2018"`.
- `-s`, `--similarity`: Specify whether to add a similarity loss in the model to order the latent space projections. Expected `type: bool`. `Default: False`.
- `-d`, `--dataset`: Specify the dataset to use. Expected `type: str`. `Default: "SleepPhysionet"`.
- `-v`, `--version`: Specify the version of the dataset. Expected `type: str`. `Default: "2018"`.
- `-c`, `--use_cache`: Specify whether to use cache for the dataset. Expected `type: bool`. `Default: True`.
- `-sl`, `--sequence_lenght`: Specify the sequence length for the model. Expected `type: int`. `Default: 3`.
- `-me`, `--max_epoch`: Specify the maximum number of epochs for training. Expected `type: int`. `Default: 20`.
- `-vci`, `--val_check_interval`: Specify the validation check interval during training. Expected `type: int`. `Default: 300`.
- `-bs`, `--batch_size`: Specify the batch size for training. Expected `type: int`. `Default: 32`.