import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer


def main():
    # Notes: With NN Potential Scaling is [-1,-2]
    # TODO: Subclass ExtrapolationVisualizer to overlay the results

    # pinn model
    df = pd.read_pickle("Data/Dataframes/earth_loss_fcn_experiment.data")
    percent_idx, rms_idx = -1, -2

    ################
    # With Percent #
    ################

    model_id = df["id"].values[percent_idx]
    config, model = load_config_and_model(model_id, df)

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 10000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(
        extrapolation_exp,
        x_axis="dist_2_COM",
        plot_fcn=plt.semilogy,
        annotate=False,
    )
    vis.fig_size = (vis.w_half, vis.w_half)
    vis.plot_interpolation_percent_error(
        plot_std=False,
        plot_max=False,
        avg_window=100,
        std_window=100,
    )
    plt.gcf().axes[0].set_ylim([1e0, 1e3])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Cost_with_Percent_Percent.pdf", pad_inches=0.0)
    plt.savefig("Plots/PINNIII/Cost_with_Percent_Percent.png", pad_inches=0.0, dpi=250)

    vis.plot_interpolation_rms(
        plot_std=False,
        plot_max=False,
        avg_window=100,
        std_window=100,
    )
    plt.gcf().axes[0].set_ylim([1e-11, 1e-3])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Cost_with_Percent_RMS.pdf", pad_inches=0.0)
    plt.savefig("Plots/PINNIII/Cost_with_Percent_RMS.png", pad_inches=0.0, dpi=250)

    ############
    # With RMS #
    ############

    model_id = df["id"].values[rms_idx]
    config, model = load_config_and_model(model_id, df)

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 10000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(
        extrapolation_exp,
        x_axis="dist_2_COM",
        plot_fcn=plt.semilogy,
        annotate=False,
    )
    vis.fig_size = (vis.w_half, vis.w_half)

    vis.plot_interpolation_percent_error(
        plot_std=False,
        plot_max=False,
        avg_window=100,
        std_window=100,
    )
    plt.gcf().axes[0].set_ylim([1e0, 1e3])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Cost_without_Percent_Percent.pdf", pad_inches=0.0)
    plt.savefig(
        "Plots/PINNIII/Cost_without_Percent_Percent.png",
        pad_inches=0.0,
        dpi=250,
    )

    vis.plot_interpolation_rms(
        plot_std=False,
        plot_max=False,
        avg_window=100,
        std_window=100,
    )
    plt.gcf().axes[0].set_ylim([1e-11, 1e-3])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Cost_without_Percent_RMS.pdf", pad_inches=0.0)
    plt.savefig("Plots/PINNIII/Cost_without_Percent_RMS.png", pad_inches=0.0, dpi=250)

    plt.show()


if __name__ == "__main__":
    main()
