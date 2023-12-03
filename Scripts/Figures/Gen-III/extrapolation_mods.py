import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer

plt.rc("text", usetex=True)


def plot(config, model, file_name, new_fig=True):
    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(
        model,
        config,
        points=1000,
        extrapolation_bound=100,
    )
    extrapolation_exp.run(override=False)

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(
        extrapolation_exp,
        x_axis="dist_2_COM",
        plot_fcn=plt.semilogy,
        annotate=False,
    )
    vis.fig_size = (vis.w_full, vis.w_half)
    if not new_fig:
        plt.figure(1)
    vis.plot_extrapolation_percent_error(
        plot_std=False,
        plot_max=False,
        new_fig=new_fig,
        annotate=False,
        linewidth=1,
        label=file_name,
    )
    plt.ylim([1e-4, 1e2])
    plt.legend()

    if not new_fig:
        plt.figure(2)
    vis.plot_interpolation_percent_error(
        plot_std=False,
        plot_max=False,
        new_fig=new_fig,
        annotate=False,
        linewidth=1,
        label=file_name,
    )
    plt.ylim([1e-4, 1e2])
    plt.legend()
    plt.tight_layout()


def main():
    # RMS #
    df = pd.read_pickle("Data/Dataframes/pinn_III_mods_PINN_II.data")
    config, model = load_config_and_model(df, idx=-1)
    plot(config, model, "PINN II")

    # RMS #
    df = pd.read_pickle("Data/Dataframes/pinn_III_mods_features.data")
    config, model = load_config_and_model(df, idx=-1)
    plot(config, model, "I: Features", new_fig=False)

    # Percent #
    df = pd.read_pickle("Data/Dataframes/pinn_III_mods_percent.data")
    config, model = load_config_and_model(df, idx=-1)
    plot(config, model, "II: Percent", new_fig=False)

    # With Scaling #
    df = pd.read_pickle("Data/Dataframes/pinn_III_mods_scaling.data")
    config, model = load_config_and_model(df, idx=-1)
    plot(config, model, "III: Scaling", new_fig=False)

    # With BC #
    df = pd.read_pickle("Data/Dataframes/pinn_III_mods_BC.data")
    config, model = load_config_and_model(df, idx=-1)
    plot(config, model, "IV: BC", new_fig=False)

    # With Fuse #
    df = pd.read_pickle("Data/Dataframes/pinn_III_mods_fuse.data")
    config, model = load_config_and_model(df, idx=-1)  #
    plot(config, model, "V: Fusing", new_fig=False)

    # plt.figure(1)
    # plt.savefig("Plots/PINNIII/PINN_III_Mods_extrapolation.pdf", pad_inches=0.0)
    # plt.savefig(
    #     "Plots/PINNIII/PINN_III_Mods_extrapolation.png",
    #     pad_inches=0.0,
    #     dpi=300,
    # )

    # plt.figure(2)
    # plt.savefig("Plots/PINNIII/PINN_III_Mods_interpolation.pdf", pad_inches=0.0)
    # plt.savefig(
    #     "Plots/PINNIII/PINN_III_Mods_interpolation.png",
    #     pad_inches=0.0,
    #     dpi=300,
    # )


if __name__ == "__main__":
    main()
    plt.show()
