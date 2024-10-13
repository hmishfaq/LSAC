import numpy as np
import torch
import random
from norm import MinMaxNormalizer
from typing import Tuple, Union, Optional

def split_diffusion_samples(
        samples: Union[np.ndarray, torch.Tensor],
        modelled_terminals: bool = True,
        terminal_threshold: Optional[float] = 0.5,
        **kwargs
    ):
    obs_dim = kwargs["obsv_dim"]
    action_dim = kwargs["action_dim"]
    # Split samples into (s, a, r, s') format
    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    next_obs = samples[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, next_obs, terminals
    else:
        return obs, actions, rewards, next_obs


class SimpleDiffusionGenerator:
    def __init__(
            self,
            ema_model,
            num_sample_steps: int = 128,
            **kwargs
        ):
        self.k = kwargs["num_parallel"]
        self.diffusion = ema_model
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.batch_size = kwargs["replay_batch_size"]

    def sample(
            self,
            networks,
            alg,
            num_samples: int,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        for _ in range(num_samples // self.batch_size):
            sampled_outputs = self.diffusion.sample(
                batch_size=self.batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
            )
            sampled_outputs = sampled_outputs.cpu().numpy()

            # Split samples into (s, a, r, s') format
            transitions = split_diffusion_samples(sampled_outputs, **kwargs)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions

            synth_pairs = {
                "syn_obs": torch.as_tensor(obs, dtype=torch.float32).to('cuda:0'),
                "syn_act": torch.as_tensor(act, dtype=torch.float32).to('cuda:0')
                }
            k = random.randint(0, self.k)
            refined_synth_pairs = alg.refine_generated_data(networks.q1[k], networks.q2[k], synth_pairs)
            obs = refined_synth_pairs["syn_obs"].cpu().numpy()
            act = refined_synth_pairs["syn_act"].cpu().numpy()

            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            terminals.append(terminal)

        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        return observations, actions, rewards, next_observations, terminals