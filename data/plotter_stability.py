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
    "OURS0": (0 / 255, 87 / 255, 183 / 255),
    "OURS1": (0 / 255, 87 / 255, 183 / 255),
    "OURS2": (0 / 255, 87 / 255, 183 / 255),
    "OURS3": (0 / 255, 87 / 255, 183 / 255),
    "OURS4": (0 / 255, 87 / 255, 183 / 255),
    "DSAC0": (255 / 255, 221 / 255, 0 / 255),
    "DSAC1": (255 / 255, 221 / 255, 0 / 255),
    "DSAC2": (255 / 255, 221 / 255, 0 / 255),
    "DSAC3": (255 / 255, 221 / 255, 0 / 255),
    "DSAC4": (255 / 255, 221 / 255, 0 / 255),
    "DIPO0": (165 / 255, 11 / 255, 94 / 255),
    "DIPO1": (165 / 255, 11 / 255, 94 / 255),
    "DIPO2": (165 / 255, 11 / 255, 94 / 255),
    "DIPO3": (165 / 255, 11 / 255, 94 / 255),
    "DIPO4": (165 / 255, 11 / 255, 94 / 255)
}
environments = [
    "stability_data/stability_ant.txt",
    "stability_data/stability_cheetah.txt",
    "stability_data/stability_hopper.txt",
    "stability_data/stability_humanoid.txt",
    "stability_data/stability_swimmer.txt",
    "stability_data/stability_walker2d.txt"
]

for file in environments:
    filename = file
    filename_without_ext = filename.rsplit('.', 1)[0]
    parts = filename_without_ext.split('_')
    env_id = parts[2]
    df = pd.read_csv(filename)
    fig, ax = plt.subplots(figsize=(8, 6))

    # List of unique algorithms
    algorithms = df['algo'].unique()
    y_max, y_min = list(), list()
    for algo in algorithms:
        # Filter data for the current algorithm
        algo_data = df[df['algo'] == algo]

        steps = algo_data['steps'].values / X_FACTOR
        rew_mean = algo_data['returns'].values
        y_max.append(max(rew_mean))
        y_min.append(min(rew_mean))
        # rew_std = algo_data['rew_std'].values

        # if algo == 'LSAC':
        #     ax.plot(steps, rew_mean, label="LSAC (ours)", linewidth=2, color=colour_list["LSAC"] + (1,))  # alpha=0.75)
        #     plt.fill_between(steps, rew_mean - rew_std, rew_mean + rew_std, facecolor=colour_list["LSAC"] + (0.3,),
        #                      linewidth=1, edgecolor=colour_list["LSAC"] + (0.4,))
        # else:
        ax.plot(steps, rew_mean, label=algo, linewidth=2, color=colour_list[algo] + (1,))  # alpha=0.75)
        # plt.fill_between(steps, rew_mean - rew_std, rew_mean + rew_std, facecolor=colour_list[algo] + (0.3,), linewidth=1,
        #              edgecolor=colour_list[algo] + (0.4,))

    # Customize the plot
    plt.xlim(0, 1)
    if file == "all_data_ant.txt":
        plt.ylim(plt.gca().get_ylim()[0], plt.gca().get_ylim()[1] + 181)
    ax.set_xlabel("Steps (million)")
    ax.yaxis.set_label_coords(Y_OFFSET, 0.5)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Average Return")
    # ax.set_title(f"{env_id}-v3")

    # Add a legend
    # fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=14)

    # Format the x-axis
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 1))
    # ax.xaxis.major.formatter._useMathText = True

    # Adjust layout
    # plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"./stability_data/{env_id}_plot.pdf", bbox_inches="tight")
    plt.show()
