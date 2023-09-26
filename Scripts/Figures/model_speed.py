import os

import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/")
    description = "no_jit"
    description = "no_jit_batch"

    pinn_df = pd.read_pickle(f"Data/Dataframes/pinn_model_times_{description}.data")
    sh_df = pd.read_pickle(f"Data/Dataframes/sh_model_times_{description}.data")

    pinn_df.sort_values(by="params", inplace=True)
    sh_df.sort_values(by="params", inplace=True)

    fig, ax = vis.newFig()
    plt.plot(pinn_df.params, pinn_df["time [s]"], label="PINN")
    plt.plot(sh_df.params, sh_df["time [s]"], label="SH")
    plt.gca().set_xscale("log", base=2)
    plt.gca().set_yscale("log", base=2)

    plt.legend()

    plt.xlabel("Parameters")
    plt.ylabel("Time [s]")

    # vis.save(fig, 'OneOff/speed_plot.pdf')
    plt.show()


if __name__ == "__main__":
    main()
