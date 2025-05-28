import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

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
    True: (254 / 255, 6 / 255, 6 / 255),
    False: (168 / 255, 47 / 255, 47 / 255),
    # "DIPO": (254 / 255, 166 / 255, 4 / 255),
    # "TD3": (6 / 255, 131 / 255, 7 / 255),
    # "PPO": (8 / 255, 156 / 255, 255 / 255),
    # "SAC": (136 / 255, 15 / 255, 135 / 255),
    # "TRPO": (129 / 255, 129 / 255, 129 / 255)
}
environments = [
    "ablat_swimmer_diffuser.txt",
]

for file in environments:
    filename = file
    filename_without_ext = filename.rsplit('.', 1)[0]
    parts = filename_without_ext.split('_')
    ablat_item = parts[1]
    df = pd.read_csv(filename)
    fig, ax = plt.subplots(figsize=(8, 6))

    # List of unique optim
    legend_handles = []
    optims = [optim for optim in df['use_diff'].unique()]
    y_max, y_min = list(), list()
    for optim in optims:
        # Filter data for the current algorithm
        algo_data = df[df['use_diff'] == optim]
        steps = algo_data['steps'].values / X_FACTOR
        rew_mean = algo_data['rew_mean'].values
        y_max.append(max(rew_mean))
        y_min.append(min(rew_mean))
        rew_std = algo_data['rew_std'].values

        legend_handles.append(Line2D([0], [0], color=colour_list[optim], lw=5, label=optim))

        ax.plot(steps, rew_mean, label=optim, linewidth=3, color=colour_list[optim] + (1,))  # alpha=0.75)
        plt.fill_between(steps, rew_mean - rew_std, rew_mean + rew_std, facecolor=colour_list[optim] + (0.3,),
                         linewidth=1,
                         edgecolor=colour_list[optim] + (0.4,))

        # plt.legend()

    # ax.legend(handles=legend_handles)

    # Customize the plot
    plt.xlim(0, 1)
    if file == "ablat_diffuser.txt":
        plt.ylim(plt.gca().get_ylim()[0], plt.gca().get_ylim()[1] + 0)
    ax.set_xlabel("Steps (million)")
    ax.yaxis.set_label_coords(Y_OFFSET, 0.5)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Average Return")
    # ax.set_title(f"{env_id}-v3")

    # Save and show the plot
    plt.savefig(f"ablation_{ablat_item}_plot.pdf", bbox_inches="tight")
    plt.show()