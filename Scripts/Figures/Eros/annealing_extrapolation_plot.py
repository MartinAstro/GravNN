import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer


class ExtrapolationVisualizerMod(ExtrapolationVisualizer):
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        plt.rc("text", usetex=True)

    def plot(self, x, value, **kwargs):
        # compute trend lines
        def get_rolling_lines(data):
            df = pd.DataFrame(data=data, index=None)
            avg = df.rolling(50, 25).mean()
            std = df.rolling(50, 25).std()
            max = df.rolling(10, 10).max()
            return avg, std, max

        # sort entries
        avg_line, std_line, max_line = get_rolling_lines(value)

        if kwargs.get("newFig", True):
            self.newFig()
        plt.scatter(x, value, alpha=0.05, s=2)

        label = kwargs.get("label", None)
        self.plot_fcn(x, avg_line, label=label)

        training_bounds = self.training_bounds / self.radius
        plt.vlines(training_bounds[0], ymin=0, ymax=np.max(value), color="green")
        plt.vlines(training_bounds[1], ymin=0, ymax=np.max(value), color="green")
        plt.vlines(1, ymin=0, ymax=np.max(value), color="grey")
        plt.tight_layout()


def format_label(config):
    constraint = config["PINN_constraint_fcn"][0]
    epochs = config["epochs"][0]
    annealing = config["lr_anneal"][0]

    constraint_str = constraint.split("_")[1].upper()

    # was the training divided between AL and A
    split_train = ""
    if epochs == 10000:
        split_train = "A+"

    annealing_str = "w/o Adaptive"
    if annealing:
        annealing_str = "w/ Adaptive"

    label = f"{split_train}{constraint_str} {annealing_str}"
    return label


def main(df_path, invert=False):
    df = pd.read_pickle(df_path)

    new_fig = True
    start_idx = -5 if invert else 0
    end_idx = -1 if invert else 4
    for i in range(start_idx, end_idx):
        model_id = df["id"].values[i]
        config, model = load_config_and_model(model_id, df)

        config["acc_noise"] = [0.0]

        # evaluate the error at "training" altitudes and beyond
        extrapolation_exp = ExtrapolationExperiment(model, config, 5000)
        extrapolation_exp.run()

        # visualize error @ training altitude and beyond
        vis = ExtrapolationVisualizerMod(
            extrapolation_exp,
            x_axis="dist_2_COM",
            plot_fcn=plt.semilogy,
        )
        vis.fig_size = vis.full_page_silver
        label = format_label(config)
        vis.plot_extrapolation_percent_error(newFig=new_fig, label=label)
        plt.gca().set_xlim([0, 10])
        plt.gca().set_ylim([1e-4, 1e1])

        new_fig = False
    plt.legend()

    # plt.show()


if __name__ == "__main__":
    main("Data/Dataframes/LR_Anneal_No_Noise_032423.data", invert=False)
    plt.gca().set_ylim([1e-3, 1e1])
    plt.savefig("Plots/PINNIII/Eros_extrapolation_LR.pdf")
    plt.savefig("Plots/PINNIII/Eros_extrapolation_LR.png", dpi=250)

    main("Data/Dataframes/LR_Anneal_With_Noise_032623.data", invert=False)
    plt.gca().set_ylim([1e-1, 1e1])
    plt.savefig("Plots/PINNIII/Eros_extrapolation_LR_noise.pdf")
    plt.savefig("Plots/PINNIII/Eros_extrapolation_LR_noise.png", dpi=250)
