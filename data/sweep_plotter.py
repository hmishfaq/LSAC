import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams["font.size"] = 30
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
rcParams["grid.color"] = "gray"
rcParams["axes.grid"] = True
rcParams["grid.linestyle"] = "--"
rcParams["grid.alpha"] = 0.3

Y_OFFSET = -0.18
X_FACTOR = 1e6
colour_list = {
    "1e-3": (254 / 255, 6 / 255, 6 / 255),
    "1e-4": (168 / 255, 47 / 255, 47 / 255),
    # "DIPO": (254 / 255, 166 / 255, 4 / 255),
    # "TD3": (6 / 255, 131 / 255, 7 / 255),
    # "PPO": (8 / 255, 156 / 255, 255 / 255),
    # "SAC": (136 / 255, 15 / 255, 135 / 255),
    # "TRPO": (129 / 255, 129 / 255, 129 / 255)
}
environments = [
    "sweep_halfcheetah_lr.txt",
]

for file in environments:
    filename = file
    filename_without_ext = filename.rsplit('.', 1)[0]
    parts = filename_without_ext.split('_')
    env_id = parts[2]
    df = pd.read_csv(filename)
    fig, ax = plt.subplots(figsize=(8, 6))

    # List of unique algorithms
    lrs = [str(lr) for lr in df['lr'].unique()]
    y_max, y_min = list(), list()
    for lr in lrs:
        # Filter data for the current algorithm
        lr = float(lr)
        algo_data = df[df['lr'] == lr]
        if lr == 0.001:
            lr = '1e-3'
        elif lr == 0.0001:
            lr = '1e-4'

        steps = algo_data['steps'].values / X_FACTOR
        rew_mean = algo_data['rew_mean'].values
        y_max.append(max(rew_mean))
        y_min.append(min(rew_mean))
        rew_std = algo_data['rew_std'].values

        ax.plot(steps, rew_mean, label=lr, linewidth=2.5, color=colour_list[lr] + (1,))  # alpha=0.75)
        plt.fill_between(steps, rew_mean - rew_std, rew_mean + rew_std, facecolor=colour_list[lr] + (0.3,), linewidth=1,
                     edgecolor=colour_list[lr] + (0.4,))
        plt.legend()

    # Customize the plot
    plt.xlim(0, 1)
    if file == "all_sweep_halfcheetah.txt":
        plt.ylim(plt.gca().get_ylim()[0], plt.gca().get_ylim()[1] + 0)
    ax.set_xlabel("Steps (million)")
    ax.yaxis.set_label_coords(Y_OFFSET, 0.5)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Policy Mean")
    # ax.set_title(f"{env_id}-v3")

    # Add a legend
    # fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=14)

    # Format the x-axis
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 1))
    # ax.xaxis.major.formatter._useMathText = True

    # Adjust layout
    # plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"sweep_{env_id}_plot.pdf", bbox_inches="tight")
    plt.show()
