__all__ = [
    "StochaPolicy",
    "ActionValueDistri",
]

import numpy as np
import warnings
import torch
import torch.nn as nn
from utils.common_utils import get_activation_func
from utils.act_distribution_cls import Action_Distribution

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers).to('cuda:0')

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    """Approximated function of stochastic policy.
    input: observation
    output: params of action distribution
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]

        assert self.std_type == "mlp_shared"
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
        self.policy = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        ).to('cuda:0')

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        assert self.std_type == "mlp_shared"
        logits = self.policy(obs.to('cuda:0'))
        action_mean, action_log_std = torch.chunk(logits, chunks=2, dim=-1)
        action_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std).exp()

        return torch.cat((action_mean, action_std), dim=-1)

class ActionValueDistri(nn.Module):
    """Approximated function of distributed action-value function.
    input: observation
    output: params of action-value distribution
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs, act):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std) # avoid 0
        
        return torch.cat((value_mean, value_log_std), dim=-1)
