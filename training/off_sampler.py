import numpy as np
import torch

from utils.initialization import create_env
from utils.common_utils import set_seed

class GaussNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, action):
        return action + np.random.normal(self.mean, self.std)

class Sampler:
    def __init__(self, index=0, **kwargs):
        self.env, *_ = create_env(**kwargs)
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 200, self.env)
        self.obs, self.info = self.env.reset()
        self.has_render = hasattr(self.env, "render")
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        self.networks = getattr(file, "Networks")(**kwargs).to('cuda:0')
        self.noise_params = kwargs["noise_params"]
        self.sample_batch_size = kwargs["batch_size_per_sampler"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.action_type = kwargs["action_type"]
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.total_sample_number = 0
        self.reward_scale = 1.0
        if self.noise_params is not None:
            self.noise_processor = GaussNoise(**self.noise_params)

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def sample(self):
        self.total_sample_number += self.sample_batch_size
        tb_info = dict()
        batch_data = []
        synth_batch_data = []
        for _ in range(self.sample_batch_size):
            batch_obs = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            )
            logits = self.networks.policy(batch_obs)

            action_distribution = self.networks.create_action_distributions(logits)
            action, logp = action_distribution.sample()
            action = action.detach()[0].cpu().numpy()
            logp = logp.detach()[0].cpu().numpy()

            if self.noise_params is not None:
                action = self.noise_processor.sample(action)

            action = np.array(action)
            action_clip = action.clip(self.env.action_space.low, self.env.action_space.high)
            next_obs, reward, self.done, next_info = self.env.step(action_clip)

            if "TimeLimit.truncated" not in next_info.keys():
                next_info["TimeLimit.truncated"] = False
            if next_info["TimeLimit.truncated"]:
                self.done = False
            data = [
                self.obs.copy(),
                self.info,
                action,
                self.reward_scale * reward,
                next_obs.copy(),
                self.done,
                logp,
                next_info,
            ]
            batch_data.append(tuple(data))

            synth_data = [
                self.obs.copy(),
                action.copy(),
                self.reward_scale * reward,
                next_obs.copy(),
                self.done,
            ]
            synth_batch_data.append(tuple(synth_data))

            self.obs = next_obs
            self.info = next_info
            if self.done or next_info["TimeLimit.truncated"]:
                self.obs, self.info = self.env.reset()
        
        return batch_data, synth_batch_data, tb_info

    def get_total_sample_number(self):
        return self.total_sample_number

def create_sampler(**kwargs):
    sampler = Sampler(**kwargs)
    return sampler
