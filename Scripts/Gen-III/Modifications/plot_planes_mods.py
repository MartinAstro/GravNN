import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

plt.rc("text", usetex=True)


def plot(config, model, file_name, new_fig=True):
    R = config["radius_max"][0]
    R = 100 * config["planet"][0].radius
    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = PlanesExperiment(
        model,
        config,
        bounds=[-R, R],
        samples_1d=100,
    )
    extrapolation_exp.run(override=False)

    # visualize error @ training altitude and beyond
    vis = PlanesVisualizer(extrapolation_exp)
    vis.fig_size = (vis.w_full, vis.w_half)
    vis.plot_percent_error(z_max=10, log=True)
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


if __name__ == "__main__":
    main()
    plt.show()
