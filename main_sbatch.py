import os
import sys
import argparse

import dmc2gym
import torch
import json
import numpy as np

from utils.initialization import create_alg, create_buffer, create_env
from training.evaluator import Evaluator
from training.off_sampler import Sampler
from training.trainer import create_trainer
from utils.sweeper import Sweeper
from utils.common_utils import change_type, seed_everything
import logging, warnings, datetime, copy

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

if __name__ == "__main__":

    #########################################################
    # Parameters Setup
    #########################################################

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--config_file', type=str, default='./configs/algo_11.json', help='choose config')
    parser.add_argument('--config_idx', type=int, default=1, help='config index')
    parser.add_argument('--slurm_dir', type=str, default='', help='slurm tempory directory')
    parser.add_argument('--run', type=int, default=1, help='number of runs')
    pre_args = parser.parse_args()

    sweeper = Sweeper(pre_args.config_file)
    cfg = sweeper.generate_config_for_idx(pre_args.config_idx)
    
    # cfg.setdefault("ini_network_dir", None)
    # cfg.setdefault("noise_params", None)
    # cfg.setdefault("save_folder", None)

    cfg['exp'] = pre_args.config_file.split('/')[-1].split('.')[0]
    local_logs_dir = f"./logs/{cfg['env']['env_type']}/{cfg['env']['env_id']}/{pre_args.config_idx}/run-{pre_args.run}/"
    make_dir(local_logs_dir)
    cfg['logs_dir'] = local_logs_dir

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"run-{pre_args.config_idx}.log"),
            logging.StreamHandler()     # Log to console
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info('sweep cfg loaded')

    #########################################################
    # Convert sweeper cfg to **args
    #########################################################

    args = {
        "env_type": cfg["env"]["env_type"],     # gym or dm-control
        "env_id": cfg["env"]["env_id"],
        "algorithm": cfg["agent"]["algo"],
        "enable_cuda": cfg["agent"]["enable_cuda"]
    }

    if cfg["env"]["generate_random_seed"]:
        args["seed"] = None
    else:
        args["seed"] = cfg["env"]["seed"]

    args["reward_scale"] = cfg["env"]["reward_scale"]
    args["action_type"] = "continu"
    args["is_render"] = False
    args["is_adversary"] = False
    args["value_func_name"] = "ActionValueDistri"

    args["value_func_type"] = cfg["value"]["value_func_type"]
    args["value_hidden_sizes"] = cfg["value"]["value_hidden_sizes"]
    args["value_hidden_activation"] = cfg["value"]["value_hidden_activation"]
    args["value_output_activation"] = cfg["value"]["value_output_activation"]
    args["value_min_log_std"] = cfg["value"]["value_min_log_std"]
    args["value_max_log_std"] = cfg["value"]["value_max_log_std"]

    args["optim"] = cfg["optimizer"]["name"]
    args["optim_lr"] = cfg["optimizer"]["kwargs"]["lr"]
    args["optim_noise_scale"] = cfg["optimizer"]["kwargs"]["noise_scale"]
    args["optim_a"] = cfg["optimizer"]["kwargs"]["a"]
    args["num_parallel"] = cfg["optimizer"]["num_parallel"]

    args["policy_func_name"] = cfg["policy"]["policy_func_name"]
    args["policy_func_type"] = cfg["policy"]["policy_func_type"]
    args["policy_act_distribution"] = cfg["policy"]["policy_act_distribution"]
    args["policy_hidden_sizes"] = cfg["policy"]["policy_hidden_sizes"]
    args["policy_hidden_activation"] = cfg["policy"]["policy_hidden_activation"]
    args["policy_output_activation"] = cfg["policy"]["policy_output_activation"]
    args["policy_min_log_std"] = cfg["policy"]["policy_min_log_std"]
    args["policy_max_log_std"] = cfg["policy"]["policy_max_log_std"]
    
    args["policy_learning_rate"] = float(cfg["policy"]["policy_learning_rate"])
    args["alpha_learning_rate"] = float(cfg["policy"]["alpha_learning_rate"])
    args["gamma"] = float(cfg["agent"]["gamma"])
    args["tau"] = float(cfg["agent"]["tau"])
    args["auto_alpha"] = cfg["agent"]["auto_alpha"]
    args["alpha"] = cfg["agent"]["alpha"]
    args["delay_update"] = cfg["policy"]["delay_update"]
    args["clip_factor"] = 3.

    args["trainer"] = cfg["sampling"]["trainer"]
    args["max_iteration"] = int(cfg["sampling"]["max_iteration"])
    args["ini_network_dir"] = None

    args["buffer_name"] = cfg["sampling"]["buffer_name"]
    args["buffer_warm_size"] = int(cfg["sampling"]["buffer_warm_size"])
    args["buffer_max_size"] = int(cfg["sampling"]["buffer_max_size"])
    args["replay_batch_size"] = int(cfg["sampling"]["replay_batch_size"])
    args["sample_interval"] = int(cfg["sampling"]["sample_interval"])
    args["sampler_name"] = cfg["sampling"]["sampler_name"]
    args["sample_batch_size"] = int(cfg["sampling"]["sample_batch_size"])
    args["upsample"] = cfg["sampling"]["upsample"]
    args["noise_params"] = None

    args["evaluator_name"] = cfg["eval"]["evaluator_name"]
    args["num_eval_episode"] = int(cfg["eval"]["num_eval_episode"])
    args["eval_interval"] = int(cfg["eval"]["eval_interval"])
    args["eval_save"] = cfg["eval"]["eval_save"]
    args["save_folder"] = cfg["logs_dir"]
    args["apprfunc_save_interval"] = int(cfg["eval"]["apprfunc_save_interval"])
    args["log_save_interval"] = int(cfg["eval"]["log_save_interval"])
    
    #########################################################
    # Initialization
    #########################################################

    seed = args.get("seed", None)
    args["seed"] = seed_everything(seed)
    env, action_scale, action_bias = create_env(**args)
    
    assert args["enable_cuda"] == True and torch.cuda.is_available()
    args["use_gpu"] = True
    args["batch_size_per_sampler"] = args["sample_batch_size"]
    args["obsv_dim"] = env.observation_space.shape[0] \
        if len(env.observation_space.shape) == 1 else env.observation_space.shape
    args["action_dim"] = env.action_space.shape[0] \
        if len(env.action_space.shape) == 1 else env.action_space.shape
    args["action_high_limit"] = env.action_space.high.astype('float32')
    args["action_low_limit"] = env.action_space.low.astype('float32')
    args["additional_info"] = {}
    args["cnn_shared"] = False

    # Save logged data
    os.makedirs(args["save_folder"], exist_ok=True)
    os.makedirs(args["save_folder"] + "/apprfunc", exist_ok=True)
    with open(args["save_folder"] + "/config.json", "w", encoding="utf-8") as f:
        json.dump(change_type(copy.deepcopy(args)), f, ensure_ascii=False, indent=4)

    args["action_scale"] = action_scale
    args["action_bias"] = action_bias

    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = Sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = Evaluator(**args)
    trainer = create_trainer(cfg, alg, sampler, buffer, evaluator, **args)

    trainer.train(**args)