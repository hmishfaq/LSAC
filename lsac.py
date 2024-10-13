__all__ = ["Networks", "LSAC"]
from copy import deepcopy
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from torch import multiprocessing
# multiprocessing.set_start_method('spawn')
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Dict, List
from utils.tensorboard_setup import tb_tags
from utils.initialization import create_apprfunc
from utils.common_utils import get_apprfunc_dict
from utils.optims import aSGLD
from training.replay_buffer import DiffusionBuffer


def lr_lambda(it, start_iter=int(1e5)):
    """Calculate learning rate based on the iteration.
    """
    return max(1.0 - 0.5 * (it - start_iter) / (int(1e6) - start_iter), 0) if it > start_iter else 1.0

class Networks(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.device = 'cuda:0'
        self.k = kwargs["num_parallel"]
        q_args = get_apprfunc_dict("value", "MLP", **kwargs)

        self.q1: List[nn.Module] = [create_apprfunc(**q_args) for _ in range(self.k)]
        self.q2: List[nn.Module] = [create_apprfunc(**q_args) for _ in range(self.k)]
        self.q1_target: List[nn.Module] = deepcopy(self.q1)
        self.q2_target: List[nn.Module] = deepcopy(self.q2)

        policy_args = get_apprfunc_dict("policy", kwargs["policy_func_type"], **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)

        for p in self.policy_target.parameters():
            p.requires_grad = False
        for i in range(self.k):
            for p in self.q1_target[i].parameters():
                p.requires_grad = False
            for p in self.q2_target[i].parameters():
                p.requires_grad = False

        self.q1_optimizer = [
            aSGLD(
                self.q1[i].parameters(),
                lr=kwargs["optim_lr"],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
                amsgrad=False,
                noise_scale=kwargs["optim_noise_scale"],
                a=kwargs["optim_a"]
            ) for i in range(self.k)
        ]
        self.q2_optimizer = [
            aSGLD(
                self.q2[i].parameters(),
                lr=kwargs["optim_lr"],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
                amsgrad=False,
                noise_scale=kwargs["optim_noise_scale"],
                a=kwargs["optim_a"]
            ) for i in range(self.k)
        ]
        self.log_alpha = nn.Parameter(
            torch.tensor(1, dtype=torch.float32)
        )
        self.policy_optimizer = Adam(
            self.policy.parameters(),
            lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam(
            [self.log_alpha],
            lr=kwargs["alpha_learning_rate"]
        )

        # Learning rate scheduler
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.policy_optimizer, lr_lambda=lr_lambda
        )
        self.q1_scheduler = [
            torch.optim.lr_scheduler.LambdaLR(
                self.q1_optimizer[i], lr_lambda=lr_lambda
            ) for i in range(self.k)
        ]
        self.q2_scheduler = [
            torch.optim.lr_scheduler.LambdaLR(
                self.q2_optimizer[i], lr_lambda=lr_lambda
            ) for i in range(self.k)
        ]

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class LSAC:

    def __init__(self, **kwargs):
        super().__init__()
        self.networks = Networks(**kwargs)
        self.device = self.networks.device
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]
        self.target_entropy = -kwargs["action_dim"]
        self.auto_alpha = kwargs["auto_alpha"]
        self.alpha = kwargs.get("alpha", 0.2)
        self.delay_update = kwargs["delay_update"]
        self.mean_std1 = -1.0
        self.mean_std2 = -1.0
        self.tau_b = kwargs.get("tau_b", self.tau)
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.clip_w = kwargs.get("clip_factor", 3.)
        self.action_gradient_steps = 1
        self.optimistic_policy = kwargs.get("optimism", False)
        self.select_best_mode = kwargs.get("best_mode", True)
        self.l2_noise_reg = kwargs.get("l2_regularization", True)
        self.noise_std = 1e-4   # std of Gaussian noise
        self.l2_lambda = 1e-4   # noise regularization parameter
        self.action_bias = kwargs["action_bias"]
        self.action_scale = kwargs["action_scale"]
        self.k = self.networks.k

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "auto_alpha",
            "alpha",
            "delay_update",
        )

    def local_update(
            self,
            data: List[Dict],
            synth_data: List[Dict],
            synth_idxs: List[np.ndarray],
            synth_ratio: float,
            synth_buffer: DiffusionBuffer,
            iteration: int) -> Dict:
        tb_info = self.grad_update(data, synth_ratio, synth_data, synth_idxs, synth_buffer, iteration)
        self.optim_update(iteration)
        return tb_info

    def compute_alpha(self, requires_grad: bool = False):
        alpha = self.networks.log_alpha.exp()
        return alpha if requires_grad else alpha.item()

    def action_gradient(self, critic1, critic2, states, unnormalized_actions):
        if isinstance(unnormalized_actions, np.ndarray):
            unnormalized_actions = torch.tensor(unnormalized_actions, device=self.device).float()
        if isinstance(self.action_bias, np.ndarray) or isinstance(self.action_scale, np.ndarray):
            action_bias = torch.tensor(self.action_bias, device=self.device).float()
            action_scale = torch.tensor(self.action_scale, device=self.device).float()

        actions = (unnormalized_actions - action_bias) / action_scale
        if not isinstance(actions, torch.nn.Parameter):
            actions = torch.nn.Parameter(actions)
        actions_optim = torch.optim.Adam([actions], lr=3e-4, eps=1e-5)

        for _ in range(self.action_gradient_steps):
            actions.requires_grad_(True)
            q1, *_ = self.compute_q(states, actions, critic1)
            q2, *_ = self.compute_q(states, actions, critic2)
            loss = - torch.min(q1, q2)

            actions_optim.zero_grad()
            loss.backward(torch.ones_like(loss))
            actions_optim.step()
            actions.requires_grad_(False)
            actions.clamp_(-1., 1.)

        new_act = actions.detach()
        with torch.no_grad():
            actions.copy_(new_act)
        unnormalized_actions = actions * action_scale + action_bias

        return states, unnormalized_actions

    def refine_synth_data(self, critic1, critic2, synth_data: Dict, synth_idxs: np.ndarray, synth_buffer: DiffusionBuffer):
        states, actions = synth_data["obs"].to(self.device), synth_data["best_act"].to(self.device)
        states, best_actions = self.action_gradient(critic1, critic2, states, actions)
        synth_buffer.replace_actions(best_actions.cpu(), synth_idxs)
        synth_data["best_act"] = best_actions

        return synth_data
    
    def refine_generated_data(self, critic1, critic2, synth_pairs: Dict):
        states, actions = synth_pairs["syn_obs"], synth_pairs["syn_act"]
        states, best_actions = self.action_gradient(critic1, critic2, states, actions)
        synth_pairs["syn_act"] = best_actions

        return synth_pairs
    
    # Multimodal posterior critic and Max-Ent policy update
    def grad_update(
            self,
            data_list: List[Dict],
            synth_ratio: float,
            synth_data_list: List[Dict],
            synth_idxs: List[np.ndarray],
            synth_buffer: DiffusionBuffer,
            iteration: int):
        q_samples = []
        s = len(next(iter((synth_data_list[0]).values())))
        diffusion_batch_size = int(s * synth_ratio)
        online_batch_size = int(s - diffusion_batch_size)

        for idx, (d1, d2) in enumerate(zip(data_list, synth_data_list)):
            self.networks.q1_optimizer[idx].zero_grad()
            self.networks.q2_optimizer[idx].zero_grad()   
            loss_q, q1, q2, std1, std2, min_std1, min_std2 = self.compute_q_loss(
                self.networks.q1[idx],
                self.networks.q2[idx],
                self.networks.q1_target[idx],
                self.networks.q2_target[idx],
                synth_idxs,
                synth_buffer,
                diffusion_batch_size,
                online_batch_size,
                d1,
                d2
            )
            loss_q.backward()
            exp_min_q = torch.exp(torch.min(q1, q2))
            q_samples.append((loss_q, q1, q2, exp_min_q, std1, std2, min_std1, min_std2))

        if self.select_best_mode:   # Optimistic multi-sampling
            exp_min_q_vals = torch.tensor([s[3] for s in q_samples])
            q_idx = torch.argmax(torch.log(exp_min_q_vals / torch.sum(exp_min_q_vals)))
        else:   # Randomly sample one Q value
            q_idx = np.random.randint(len(data_list))
        (loss_q, q1, q2, exp_min_q, std1, std2, min_std1, min_std2) = q_samples[q_idx]

        obs = data_list[q_idx]["obs"].to(self.device)
        logits = self.networks.policy(obs)
        logits_mean, logits_std = torch.chunk(logits, chunks=2, dim=-1)
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()

        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_log_prob = act_dist.rsample()
        data_list[q_idx].update({"new_act": new_act, "new_log_prob": new_log_prob})

        for p in self.networks.q1[q_idx].parameters():
            p.requires_grad = False
        for p in self.networks.q2[q_idx].parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy = self.optimistic_policy_loss(
            self.networks.q1, self.networks.q2, data_list[q_idx]
        ) if self.optimistic_policy else self.compute_loss_policy(
            self.networks.q1[q_idx], self.networks.q2[q_idx], data_list[q_idx]
        )
        loss_policy.backward()

        for p in self.networks.q1[q_idx].parameters():
            p.requires_grad = True
        for p in self.networks.q2[q_idx].parameters():
            p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            new_log_prob = data_list[q_idx]["new_log_prob"]
            loss_alpha = - self.networks.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
            loss_alpha.backward()

        tb_info = {
            "critic_avg_q1": q1.item(),
            "critic_avg_q2": q2.item(),
            "critic_avg_std1": std1.item(),
            "critic_avg_std2": std2.item(),
            "critic_avg_min_std1": min_std1.item(),
            "critic_avg_min_std2": min_std2.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: loss_q.item(),
            "policy_mean": policy_mean,
            "policy_std": policy_std,
            "entropy": entropy.item(),
            "alpha": self.compute_alpha(),
            "mean_std1": self.mean_std1,
            "mean_std2": self.mean_std2,
        }

        return tb_info

    def compute_q(self, obs, act, qnet):
        StochaQ = qnet(obs.to(self.device), act.to(self.device))
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -self.clip_w, self.clip_w)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def compute_q_loss(self, critic1, critic2, q1_target, q2_target, idxs, synth_buffer, d_size, o_size, data, synth_data):
        self.refine_synth_data(critic1, critic2, synth_data, idxs, synth_buffer)
        if d_size < o_size:
            data["obs"] = torch.Tensor(
                np.concatenate(
                    [synth_data["obs"][:d_size], data["obs"][d_size:]], axis=0
                )
            )
            data["act"] = torch.Tensor(
                np.concatenate(
                    [synth_data["best_act"][:d_size], data["act"][d_size:]], axis=0
                )
            )
            data["rew"] = torch.Tensor(
                np.concatenate(
                    [synth_data["rew"][:d_size], data["rew"][d_size:]], axis=0
                )
            )
            data["obs2"] = torch.Tensor(
                np.concatenate(
                    [synth_data["obs2"][:d_size], data["obs2"][d_size:]], axis=0
                )
            )
            data["done"] = torch.Tensor(
                np.concatenate(
                    [synth_data["done"][:d_size], data["done"][d_size:]], axis=0
                )
            )
        obs, act, rew, obs2, done = (
            data["obs"].to(self.device).clone().detach(),
            data["act"].to(self.device).clone().detach(),   # best_act
            data["rew"].to(self.device).clone().detach(),
            data["obs2"].to(self.device).clone().detach(),
            data["done"].to(self.device).clone().detach(),
        )
        logits_2 = self.networks.policy_target(obs2)
        act2_dist = self.networks.create_action_distributions(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()

        q1, q1_std, _ = self.compute_q(obs, act, critic1)
        q2, q2_std, _ = self.compute_q(obs, act, critic2)

        self.mean_std1 = torch.mean(q1_std.detach()) if self.mean_std1 == -1.0 \
            else (1 - self.tau_b) * self.mean_std1 + self.tau_b * torch.mean(q1_std.detach())
        self.mean_std2 = torch.mean(q2_std.detach()) if self.mean_std2 == -1.0 \
            else (1 - self.tau_b) * self.mean_std2 + self.tau_b * torch.mean(q2_std.detach())

        q1_next, _, q1_next_sample = self.compute_q(obs2, act2, q1_target)
        q2_next, _, q2_next_sample = self.compute_q(obs2, act2, q2_target)
        q_next = torch.min(q1_next, q2_next)
        q_next_sample = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)

        # Compute target_q and q_clip
        target_q = rew + (1 - done) * self.gamma * (
            q_next.detach() - self.compute_alpha() * log_prob_act2.detach()
        ).detach()
        target_q_sample = rew + (1 - done) * self.gamma * (
            q_next_sample.detach() - self.compute_alpha() * log_prob_act2.detach()
        )
        clip_q1_bound = self.clip_w * self.mean_std1.detach()
        clip_q1 = q1.detach() + torch.clamp(target_q_sample - q1.detach(), -clip_q1_bound, clip_q1_bound)
        clip_q2_bound = self.clip_w * self.mean_std2.detach()
        clip_q2 = q2.detach() + torch.clamp(target_q_sample - q2.detach(), -clip_q2_bound, clip_q2_bound)

        q1_std_detach = torch.clamp(q1_std, min=0.).detach()
        q2_std_detach = torch.clamp(q2_std, min=0.).detach()

        q1_loss = self.distributed_q_loss(self.mean_std1, target_q, q1, q1_std_detach, clip_q1)
        q2_loss = self.distributed_q_loss(self.mean_std2, target_q, q2, q2_std_detach, clip_q2)
        
        # L2 regularization of Q loss
        if self.l2_noise_reg:
            p1_l2_reg = torch.tensor(0.0, device=self.device)
            p2_l2_reg = torch.tensor(0.0, device=self.device)

            for p1, p2 in zip(critic1.parameters(), critic2.parameters()):
                p1_noise = torch.tensor(
                    torch.randn(p1.size()) * self.noise_std,
                    device=self.device
                )
                p2_noise = torch.tensor(
                    torch.randn(p2.size()) * self.noise_std,
                    device=self.device
                )
                p1_l2_reg += torch.linalg.norm(p1 + p1_noise)
                p2_l2_reg += torch.linalg.norm(p2 + p2_noise)
            
            q1_loss += self.l2_lambda * p1_l2_reg
            q2_loss += self.l2_lambda * p2_l2_reg

        return q1_loss + q2_loss, q1.detach().mean(), q2.detach().mean(), \
            q1_std.detach().mean(), q2_std.detach().mean(), q1_std.min().detach(), q2_std.min().detach()

    def distributed_q_loss(self, mean_std, target_q, q, q_std, q_clip, b: float = 0.1):
        """Calculate variance-based distributed critic loss.
        """
        mean_grad = - (target_q - q).detach() / (q_std.pow(2) + b) * q
        var_grad = - ((q.detach() - q_clip).pow(2) - q_std.pow(2)) / (q_std.pow(3) + b) * q_std
        moving_avg = mean_std.pow(2) + b

        return moving_avg * torch.mean(mean_grad + var_grad)

    def optimistic_policy_loss_softmax(self, q1_list, q2_list, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        q_vals, exp_q_vals = [], []
        obs = data["obs"].to(self.device)
        new_act = data["new_act"].to(self.device)
        new_log_prob = data["new_log_prob"].to(self.device)

        for critic1, critic2 in zip(q1_list, q2_list):
            q1 = self.compute_q(obs, new_act, critic1)[0]
            q2 = self.compute_q(obs, new_act, critic2)[0]
            min_q = torch.min(q1, q2)
            q_vals.append(min_q)
            exp_q_vals.append(torch.exp(min_q.mean()))

        exp_q_vals = torch.tensor(exp_q_vals)
        normalized_exp_q_vals = exp_q_vals / torch.sum(exp_q_vals)
        log_normalized_exp_q_vals = torch.log(normalized_exp_q_vals)
        max_qi = torch.argmax(log_normalized_exp_q_vals)
        q_max = q_vals[max_qi]

        loss_policy = (self.compute_alpha() * new_log_prob - q_max).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def optimistic_policy_loss(self, q1_list, q2_list, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = data["obs"].to(self.device)
        new_act = data["new_act"].to(self.device)
        new_log_prob = data["new_log_prob"].to(self.device)

        q_max = torch.full_like(
            self.compute_q(obs, new_act, q1_list[0])[0],
            float('-inf')
        )
        for critic1, critic2 in zip(q1_list, q2_list):
            q1 = self.compute_q(obs, new_act, critic1)[0]
            q2 = self.compute_q(obs, new_act, critic2)[0]
            q = torch.min(q1, q2)
            q_max = torch.max(q_max, q)
        
        loss_policy = (self.compute_alpha() * new_log_prob - q_max).mean()
        entropy = - new_log_prob.detach().mean()
        return loss_policy, entropy

    def compute_loss_policy(self, q1, q2, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, new_act, new_log_prob = data["obs"].to(self.device), data["new_act"].to(self.device), data["new_log_prob"].to(self.device)
        q1, *_ = self.compute_q(obs, new_act, q1)
        q2, *_ = self.compute_q(obs, new_act, q2)
        loss_policy = (self.compute_alpha() * new_log_prob - torch.min(q1,q2)).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def optim_update(self, iteration: int):
        for i in range(self.k):
            self.networks.q1_optimizer[i].step()
            self.networks.q2_optimizer[i].step()
            self.networks.q1_scheduler[i].step()
            self.networks.q2_scheduler[i].step()
        self.networks.policy_scheduler.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()
            self.networks.alpha_optimizer.step()
            with torch.no_grad():
                polyak = 1 - self.tau
                for i in range(self.k):
                    for p, p_targ in zip(
                        self.networks.q1[i].parameters(),
                        self.networks.q1_target[i].parameters()
                    ):
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)
                    for p, p_targ in zip(
                        self.networks.q2[i].parameters(),
                        self.networks.q2_target[i].parameters()
                    ):
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.networks.policy.parameters(),
                    self.networks.policy_target.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
