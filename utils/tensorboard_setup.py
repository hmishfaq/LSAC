import numpy as np
import pandas as pd
import os
import tensorboard
import tensorboard.backend.application
from tensorboard.backend.event_processing import event_accumulator

tensorboard.backend.application.logger.setLevel("ERROR")


def add_scalars(tb_info, writer, step):
    for key, value in tb_info.items():
        writer.add_scalar(key, value, step)

def save_data(cfg, seed, returns, step):

    csv_dir = cfg["logs_dir"]
    csv_path = os.path.join(csv_dir, "{}_{}.csv".format('returns', str(seed)))
    df = pd.DataFrame([{
        "seed": seed,
        "environment": cfg["env"]["env_id"],
        "agent": cfg["agent"]["algo"],
        "config": cfg["config_idx"],
        "step": step,
        "return": returns,
        "optimizer/name": cfg["optimizer"]["name"],
        "optimizer/lr": cfg["optimizer"]["kwargs"]["lr"],
        "optimizer/noise_scale": cfg["optimizer"]["kwargs"]["noise_scale"],
        "optimizer/a": cfg["optimizer"]["kwargs"]["a"],
        }])
    df.to_csv(csv_path,
              mode='a',
              header=not os.path.isfile(csv_path),
              index=False,
              sep=",")

tb_tags = {
    "total_avg_ret": "total average return",
    "loss_actor": "actor loss",
    "loss_critic": "critic loss",
}