import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer


def plot(config, model, name):
    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(
        model,
        config,
        10000,
        extrapolation_bound=100,
    )
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(
        extrapolation_exp,
        x_axis="dist_2_COM",
        plot_fcn=plt.plot,
        annotate=False,
    )
    vis.fig_size = (vis.w_half, vis.w_half)
    vis.plot_interpolation_percent_error(
        plot_std=False,
        plot_max=False,
        avg_window=250,
        std_window=100,
        annotate=False,
        line_color="grey",
        linewidth=2,
    )
    plt.gcf().axes[0].set_ylim([0, 1.0])
    plt.tight_layout()
    vis.save(plt.gcf(), f"{name}_Percent")

    # vis.plot_interpolation_rms(
    #     plot_std=False,
    #     plot_max=False,
    #     avg_window=100,
    #     std_window=100,
    # )
    # plt.gcf().axes[0].set_ylim([1e-11, 1e-3])
    # plt.tight_layout()
    # vis.save(plt.gcf(), f"{name}_RMS")


def main():
    # TODO: Subclass ExtrapolationVisualizer to overlay the results

    df = pd.read_pickle("Data/Dataframes/pinn_loss_mod_I.data")
    config, model = load_config_and_model(df)
    plot(config, model, "Cost_RMS")

    df = pd.read_pickle("Data/Dataframes/pinn_loss_mod_II.data")
    config, model = load_config_and_model(df)
    plot(config, model, "Cost_RMS_Percent")

    plt.show()


if __name__ == "__main__":
    main()
