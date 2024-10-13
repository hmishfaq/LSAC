__all__ = ["Trainer"]

from cmath import inf
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from utils.tensorboard_setup import add_scalars
from utils.tensorboard_setup import tb_tags, save_data
from utils.common_utils import ModuleOnDevice

from replay_buffer import ReplayBuffer, DiffusionBuffer
from denoiser_network import ResidualMLPDenoiser
from elucidated_diffusion import ElucidatedDiffusion, Trainer, DiffuserTrainer
from norm import normalizer_factory, MinMaxNormalizer
from train_diffuser import SimpleDiffusionGenerator

from typing import List, Tuple, Optional

class Trainer:
    def __init__(self, cfg, alg, sampler, buffer, evaluator, **kwargs):
        self.cfg = cfg
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.synth_buffer = DiffusionBuffer(**kwargs)
        self.k = kwargs["num_parallel"]
        self.synth_ratio = kwargs.get("ratio", 0.5)
        self.replay_batch_size = kwargs["replay_batch_size"]
        self.trainer = kwargs["trainer"]
        self.seed = kwargs["seed"]
        self.evaluator = evaluator

        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        self.max_iteration = kwargs["max_iteration"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0

        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.writer.flush()

        # pre sampling
        while self.buffer.size < kwargs["buffer_warm_size"]:
            samples, synth_samples, _ = self.sampler.sample()
            self.buffer.add_batch(samples)
            self.synth_buffer.add_batch(synth_samples)

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()

        # Diffusion generator
        self.disable_diffuser = not kwargs["upsample"]
        self.retrain_diffuser_every = int(1e4)
        self.model_terminals = True
        self.diff_dims = kwargs["obsv_dim"] + kwargs["action_dim"] + 1 + kwargs["obsv_dim"] + 1
        self.skip_reward_norm = False   # NOTE: For DMC this could be set to True

    def construct_diffuser(
        self,
        inputs: torch.Tensor,
        normalizer_type: str = 'standard',
        denoising_network: nn.Module = ResidualMLPDenoiser,
        disable_terminal_norm: bool = True,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = None,
    ) -> ElucidatedDiffusion:

        event_dim = inputs.shape[1]
        model = denoising_network(d_in=event_dim, cond_dim=cond_dim)

        if disable_terminal_norm:
            terminal_dim = event_dim - 1
            if terminal_dim not in skip_dims:
                skip_dims.append(terminal_dim)

        normalizer = normalizer_factory(normalizer_type, inputs, skip_dims=skip_dims)

        return ElucidatedDiffusion(
            net=model,
            normalizer=normalizer,
            event_shape=[event_dim],
        )

    
    def step(self, **kwargs):
        # sampling
        sampler_tb_dict = {}
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cuda:0"):
                sampler_samples, synth_sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
            self.synth_buffer.add_batch(synth_sampler_samples)

        # replay
        replay_samples_list = []
        synth_replay_samples_list = []
        synth_idxs_list = []

        for _ in range(self.k):
            replay_samples, _ = self.buffer.sample_batch(self.replay_batch_size)
            replay_samples_list.append(replay_samples)
            synth_replay_samples, synth_idxs = self.synth_buffer.sample_batch(self.replay_batch_size)
            synth_replay_samples_list.append(synth_replay_samples)
            synth_idxs_list.append(synth_idxs)
    
        inputs = torch.zeros((128, self.diff_dims)).float()
        skip_dims = [kwargs["obsv_dim"] + kwargs["action_dim"]] if self.skip_reward_norm else []

        if not self.disable_diffuser and (self.iteration + 1) % self.retrain_diffuser_every == 0:
            print('retraining diffuser')
            synth_trainer = DiffuserTrainer(
                self.construct_diffuser(
                    inputs=inputs,
                    skip_dims=skip_dims,
                    disable_terminal_norm=self.model_terminals
                ),
                train_num_steps=int(1e5),
                model_terminals=self.model_terminals
            )
            synth_trainer.update_normalizer(self.buffer)
            synth_trainer.train_from_replay_buffer(self.buffer)
            # self.synth_buffer = DiffusionBuffer(**kwargs)        # optional to reset synth_buffer

            synth_data = []
            generator = SimpleDiffusionGenerator(
                ema_model=synth_trainer.ema.ema_model,
                num_sample_steps=128,
                **kwargs
            )
            obs, act, rew, obs2, done = generator.sample(
                networks=self.networks,
                alg=self.alg,
                num_samples=self.replay_batch_size,
                **kwargs
            )

            for o, a, r, o2, d in zip(obs, act, rew, obs2, done):
                synth_data.append(tuple([o, a, r, o2, d]))
            self.synth_buffer.add_batch(synth_data)

        alg_tb_dict = self.alg.local_update(
            replay_samples_list,
            synth_replay_samples_list,
            synth_idxs_list,
            self.synth_ratio,
            self.synth_buffer,
            self.iteration
        )

        # log
        if self.iteration % self.log_save_interval == 0:
            print('iteration {}'.format(self.iteration))
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

        # evaluate
        if self.iteration % self.eval_interval == 0:
            with ModuleOnDevice(self.networks, "cuda:0"):
                total_avg_return = self.evaluator.run_evaluation(self.iteration)

            if (
                total_avg_return >= self.best_tar
                and self.iteration >= self.max_iteration / 5
            ):
                self.best_tar = total_avg_return

                for filename in os.listdir(self.save_folder + "/apprfunc/"):
                    if filename.endswith("_opt.pkl"):
                        os.remove(self.save_folder + "/apprfunc/" + filename)

                torch.save(
                    self.networks.state_dict(),
                    self.save_folder
                    + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                )

            self.writer.add_scalar(tb_tags["total_avg_ret"], total_avg_return, self.iteration)
            save_data(self.cfg, self.seed, total_avg_return, self.iteration)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

    def train(self, **kwargs):
        while self.iteration < self.max_iteration:
            self.step(**kwargs)
            self.iteration += 1
            print(self.iteration)

        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )


def create_trainer(cfg, alg, sampler, buffer, evaluator, **kwargs):
    trainer = Trainer(cfg, alg, sampler, buffer, evaluator, **kwargs)
    return trainer