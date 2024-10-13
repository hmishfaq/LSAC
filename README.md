# Langevin Soft Actor Critic (LSAC)

PyTorch implementation of the *Langevin Soft Actor Critic (LSAC)* algorithm.

## Installation

Please install the following libraries and dependencies in a conda environment:

```bash
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
module load anaconda/3
# make sure you are in the cloned directory
conda env create -p lsac_condaenv python=3.10 
conda activate ./lsac_condaenv
pip install -r requirements.txt
```

Please double check if MuJoCo is properly installed and GPU training is enabled:

```bash
# mujoco and mujoco-py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/home/dir/.mujoco/mujoco210/bin
# nvidia driver
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

For DeepMind Control experiments specifically, please install these additional requirements:

```bash
# dm-control
pip install git+https://github.com/google-deepmind/dm_control.git
# dmc2gym
pip install -e git+https://github.com/denisyarats/dmc2gym.git@06f7e335d988b17145947be9f6a76f557d0efe81#egg=dmc2gym
```

## Experiments

The best tuned hyperparameters set including the tested environments are stored in a configuration file in directory `configs/`. There are two configs, one for MuJoCo Gym and one for DeepMind Control Suite experiments.

To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results including checkpoints, tensorboard files, and return csv data are saved in directory `logs/<env_type>/<env_id>/<cfg>/run-<seed>/`. You can view and analyze results directly by `tensorboard --logdir <log_dir>` or use `head logs/<path_to_csv>` to view average returns for each `seed` and `env_id`.

As an example, to run our algorithm with Gym `Ant-v3` from `configs/gym_exp.json`, we can use configuration index `1`:

```bash
python main_sbatch.py --config_file ./configs/gym_exp.json --config_idx 1
```

The number of total combinations (environments) in a config file can be calculated by running

```bash
python utils/sweeper.py
```

Then we run through all configuration indexes from `1` to `<num_envs>`, using a bash script:

``` bash
for index in {1..6}
do
  python main_sbatch.py --config_file ./configs/gym_exp.json --config_idx $index
done
```

It is also possible to run multiple seeds for all environments. [Parallel](https://www.gnu.org/software/parallel/) is usually a better choice to schedule such a large number of jobs. For example, if we want to test across ten random seeds, we could simply do:

``` bash
parallel --eta --ungroup python main_sbatch.py --config_file ./configs/gym_exp.json --config_idx {1} ::: $(seq 1 60)
```

Any configuration index that has the same remainder (divided by the number of total combinations) should have the same configuration dict. So for multiple runs, we just need to add the number of total combinations to the configuration index. For example, five runs for `Ant-v3` (config index `1`):

```bash
parallel --eta --ungroup python main_sbatch.py --config_file ./configs/gym_exp.json --config_idx {1} ::: $(seq 1 6 30)
```