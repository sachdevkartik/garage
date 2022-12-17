import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd
import argparse
import numpy as np
import os
import matplotlib
import yaml
from yaml.loader import SafeLoader

matplotlib.use("TkAgg")
import seaborn as sns

sns.set()

COLORS = [
    # deepmind style
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    # built-in color
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "purple",
    "pink",
    "brown",
    "orange",
    "teal",
    "lightblue",
    "lime",
    "lavender",
    "turquoise",
    "darkgreen",
    "tan",
    "salmon",
    "gold",
    "darkred",
    "darkblue",
    # personal color
    "#313695",  # DARK BLUE
    "#74add1",  # LIGHT BLUE
    "#4daf4a",  # GREEN
    "#f46d43",  # ORANGE
    "#d73027",  # RED
    "#984ea3",  # PURPLE
    "#f781bf",  # PINK
    "#ffc832",  # YELLOW
    "#000000",  # BLACK
]


class Plotter:
    def __init__(self, env) -> None:
        # plotting mean and std
        self.fig, self.ax = plt.subplots(ncols=1, sharey=True)
        self.env = env
        pass

    def moving_average(self, x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    def tsplot(
        self,
        ax,
        env_steps,
        avg_return,
        all_min_return,
        all_max_return,
        sliding_window=10,
        color_fill=COLORS[0],
        color_solid=COLORS[4],
        label="random_label",
        **kw,
    ):
        x = np.array(env_steps)
        est = np.mean(avg_return, axis=0)
        min_mean = np.min(all_min_return, axis=0)
        max_mean = np.max(all_max_return, axis=0)

        # sd = np.std(avg_return, axis=0)
        # cis = (est - sd, est + sd)

        cis = (min_mean, max_mean)

        # compute exponential moving average
        sliding_window = sliding_window
        est_ema = self.moving_average(est, w=sliding_window)
        modified_time_steps = np.shape(env_steps)[0] - sliding_window + 1

        # plot min and max returns
        ax.fill_between(
            x[:modified_time_steps],
            cis[0][:modified_time_steps],
            cis[1][:modified_time_steps],
            alpha=0.2,
            facecolor=color_fill,
            **kw,
        )

        # plot avg return
        ax.plot(x[:modified_time_steps], est_ema, color=color_solid, label=label, **kw)
        ax.margins(x=0)

    def data_gen(self, avg_return):
        avg_return_noisy1 = avg_return - 100 * np.random.normal(
            0, 1, np.size(avg_return)
        )
        avg_return_noisy2 = avg_return + 40 * np.random.normal(
            0, 1, np.size(avg_return)
        )

        return avg_return_noisy1, avg_return_noisy2

    def test_plotting(self, logging_data_path):
        logging_data_path = logging_data_path
        logging_data = pd.read_csv(logging_data_path)

        # define x and y
        env_steps = logging_data["TotalEnvSteps"]
        avg_return = logging_data["Evaluation/AverageReturn"]
        min_return = logging_data["Evaluation/MinReturn"]
        max_return = logging_data["Evaluation/MaxReturn"]

        # fake data generation for avg, min and max
        avg_return_noisy1, avg_return_noisy2 = self.data_gen(avg_return)
        min_return_noisy1, min_return_noisy2 = self.data_gen(min_return)
        max_return_noisy1, max_return_noisy2 = self.data_gen(max_return)

        all_avg_return = np.vstack((avg_return, avg_return_noisy1, avg_return_noisy2))
        all_min_return = np.vstack((min_return, min_return_noisy1, min_return_noisy2))
        all_max_return = np.vstack((max_return, max_return_noisy1, max_return_noisy2))

        self.tsplot(
            self.ax,
            env_steps,
            all_avg_return,
            all_min_return,
            all_max_return,
            sliding_window=10,
            color_fill=COLORS[5],
            color_solid=COLORS[1],
        )

        self.plot()

    def plot(self):
        # decoration
        self.ax.set_title(self.env)
        self.ax.set_xlabel("million steps")
        self.ax.set_ylabel("average return")  #
        self.ax.legend(loc="best", frameon=True, fancybox=True)

        self.fig.tight_layout()
        plt.show()

    def plot_eval_results(
        self, hyperparameter_to_analyze, seeds, batch_size_list, log_path
    ):
        # plot for respective hyperparameter
        assert hyperparameter_to_analyze is not None, "specify which parameter to plot"

        if hyperparameter_to_analyze == "batch_size":
            labels = []
            for i, batch_size in enumerate(batch_size_list):
                labels.append(batch_size)
                color_patch = []

                all_avg_return, all_min_return, all_max_return = [], [], []
                for seed in seeds:
                    progress_filename = (
                        f"sac_batch_size_{str(batch_size)}_{seed}/progress.csv"
                    )
                    progress_filepath = os.path.join(log_path, progress_filename)

                    # load pands
                    logging_data = pd.read_csv(progress_filepath)

                    # define x and y
                    env_steps = logging_data["TotalEnvSteps"]
                    avg_return = logging_data["Evaluation/AverageReturn"]
                    min_return = logging_data["Evaluation/MinReturn"]
                    max_return = logging_data["Evaluation/MaxReturn"]

                    all_avg_return.append(avg_return)
                    all_min_return.append(min_return)
                    all_max_return.append(max_return)

                # define color for the batch size
                color_patch.append(mpatches.Patch(color=COLORS[i], label=batch_size))

                all_avg_return = np.vstack(all_avg_return)
                all_min_return = np.vstack(all_min_return)
                all_max_return = np.vstack(all_max_return)

                self.tsplot(
                    self.ax,
                    env_steps,
                    all_avg_return,
                    all_min_return,
                    all_max_return,
                    sliding_window=1,
                    color_fill=COLORS[i],
                    color_solid=COLORS[i],
                    label=batch_size,
                )

            # plt.legend(
            #     frameon=True, fancybox=True, handles=color_patch, loc="best",
            # )

            self.plot()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logging_data_path", default=None, help="path to logging data"
    )
    parser.add_argument(
        "--test_plotting", action="store_true", help="test plotting or not"
    )

    parser.add_argument(
        "--hyperparameter_to_analyze",
        help="hyperparameter to analyze",
        default=None,
        choices=[
            "batch_size",
            "layer_normalization",
            "activation_functions",
            "policy_net",
            "q_function_net",
        ],
    )

    parser.set_defaults(test_plotting=False)

    args = parser.parse_args()

    # load benchmark config data
    benchmark_config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmark_config.yml",
    )

    hyperparameter_to_analyze = args.hyperparameter_to_analyze

    # Open the file and load the file
    with open(benchmark_config_file) as f:
        benchmark_config_data = yaml.load(f, Loader=SafeLoader)

    env = benchmark_config_data["env"]
    seeds = benchmark_config_data["seeds"]
    algo = benchmark_config_data["algo"]
    batch_size_list = benchmark_config_data["batch_size_list"]
    epochs = benchmark_config_data["epochs"]
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/local/experiment"
    )

    # define plotter
    plotter = Plotter(env)

    # test plotter
    if args.test_plotting:
        plotter.test_plotting(args.logging_data_path)

    plotter.plot_eval_results(
        hyperparameter_to_analyze=hyperparameter_to_analyze,
        seeds=seeds,
        batch_size_list=batch_size_list,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
