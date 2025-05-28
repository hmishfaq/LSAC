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

Y_OFFSET = -0.15
X_FACTOR = 1e6
colour_list = {
    "DMLMC (ours)": (69 / 255, 63 / 255, 232 / 255),
    "DSAC-T": (41 / 255, 132 / 255, 81 / 255),
    "DIPO": (121 / 255, 21 / 255, 121 / 255),
    "TD3": (210 / 255, 116 / 255, 7 / 255),
    "SAC": (90 / 255, 145 / 255, 172 / 255),
    "PPO": (154 / 255, 0 / 255, 255 / 255),
    "TRPO": (129 / 255, 129 / 255, 129 / 255)
}
environments = [
    "state_coverage_antmaze.txt"
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
        rew_mean = algo_data['rew_mean'].values
        y_max.append(max(rew_mean))
        y_min.append(min(rew_mean))
        rew_std = algo_data['rew_std'].values

        if algo == 'OURS':
            ax.plot(steps, rew_mean, label="DMLMC (ours)", linewidth=2, color=colour_list["DMLMC (ours)"] + (1,), marker='o', markersize=8)  # alpha=0.75)
            print(rew_mean, rew_std)
            plt.fill_between(steps, rew_mean - rew_std, rew_mean + rew_std, facecolor=colour_list["DMLMC (ours)"] + (0.3,),
                             linewidth=1, edgecolor=colour_list["DMLMC (ours)"] + (0.4,))
        elif algo == 'DSAC':
            ax.plot(steps, rew_mean, label="DSAC-T", linewidth=2,
                    color=colour_list["DSAC-T"] + (1,), marker='o', markersize=8)  # alpha=0.75)
            plt.fill_between(steps, rew_mean - rew_std, rew_mean + rew_std,
                             facecolor=colour_list["DSAC-T"] + (0.3,),
                             linewidth=1, edgecolor=colour_list["DSAC-T"] + (0.4,))
        else:
            ax.plot(steps, rew_mean, label=algo, linewidth=2, color=colour_list[algo] + (1,), marker='o', markersize=8)  # alpha=0.75)
            plt.fill_between(steps, rew_mean - rew_std, rew_mean + rew_std, facecolor=colour_list[algo] + (0.3,), linewidth=1,
                         edgecolor=colour_list[algo] + (0.4,))

    # Customize the plot
    plt.xlim(0, 1)
    if file == "all_data_ant.txt":
        plt.ylim(plt.gca().get_ylim()[0], plt.gca().get_ylim()[1] + 181)
    if file == 'state_coverage_antmaze.txt' or file == 'state_coverage_custom_pointmaze.txt':
        ax.set_ylim(-0.03, 1.00)
    ax.set_xlabel("Steps (million)")
    ax.yaxis.set_label_coords(Y_OFFSET, 0.5)
    plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_ylabel("State Coverage")
    # ax.set_title(f"{env_id}-v3")

    # Add a legend
    # fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=14)

    # Format the x-axis
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 1))
    # ax.xaxis.major.formatter._useMathText = True

    # Adjust layout
    # plt.tight_layout()

    # Save and show the plot
    # plt.tight_layout()
    plt.savefig(f"{env_id}_state_coverage.pdf", bbox_inches="tight")
    plt.show()
