import importlib
import os.path
import sys

import gym
import dmc2gym
from dm_control import suite
from wrapping_env import wrapping_env
from typing import Optional, Tuple, Dict, Any


def create_env(**kwargs: Dict[str, Any]) -> Tuple[gym.Env, float, float]:

    env_type = kwargs["env_type"]       # gym or dm-control
    env_name = kwargs["env_id"]
    seed = kwargs["seed"]

    if env_type == 'gym':
        env_specs = gym.envs.registry.all()
        env_ids = [spec.id for spec in env_specs]
        if env_name not in env_ids:
            raise NotImplementedError("env is not properly defined")
        try:
            env = gym.make(env_name)
        except:
            raise ModuleNotFoundError("mujoco, mujoco-py, or MSVC is not installed")
        action_scale = (env.action_space.high - env.action_space.low) / 2.
        action_bias = (env.action_space.high + env.action_space.low) / 2.

    elif env_type == 'dm-control':
        max_len = max(len(d) for d, _ in suite.BENCHMARKING)
        # for domain, task in suite.BENCHMARKING:
        #     print(f'{domain:<{max_len}}  {task}')
        domain, task = env_name.split('-')
        try:
            env = dmc2gym.make(domain_name=domain, task_name=task, seed=seed)
        except Exception as e:
            raise NotImplementedError(f"failed to create environment: {e}")
        action_scale = (env.action_space.high - env.action_space.low) / 2.
        action_bias = (env.action_space.high + env.action_space.low) / 2.

    max_episode_steps = kwargs.get("max_episode_steps", None)
    reward_scale = kwargs.get("reward_scale", None)
    reward_shift = kwargs.get("reward_shift", None)
    env = wrapping_env(
        env_type=env_type,
        env=env,
        max_episode_steps=max_episode_steps,
        reward_shift=reward_shift,
        reward_scale=reward_scale,
    )

    return env, action_scale, action_bias


def create_alg(**kwargs):
    alg_name = kwargs["algorithm"]
    alg_file_name = alg_name.lower()
    module = importlib.import_module(alg_file_name)
    assert hasattr(module, alg_name), "algorithm is not properly defined"
    algo = getattr(module, alg_name)(**kwargs)

    return algo

def create_apprfunc(**kwargs):
    apprfunc_name = kwargs["apprfunc"]
    apprfunc_file_name = apprfunc_name.lower()
    try:
        file = importlib.import_module('networks.' + apprfunc_file_name)
    except NotImplementedError:
        raise NotImplementedError("this apprfunc does not exist")

    name = formatter(kwargs["name"])

    if hasattr(file, name):
        apprfunc_cls = getattr(file, name)
        apprfunc = apprfunc_cls(**kwargs)
    else:
        raise NotImplementedError("this apprfunc is not properly defined")

    return apprfunc

def create_buffer(**kwargs):
    buffer_file_name = kwargs["buffer_name"].lower()
    module = importlib.import_module("training." + buffer_file_name)
    buffer_name = formatter(buffer_file_name)

    if hasattr(module, buffer_name):
        buffer_cls = getattr(module, buffer_name)
        buffer = buffer_cls(**kwargs)
    else:
        raise NotImplementedError("this buffer is not properly defined")

    return buffer


def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
