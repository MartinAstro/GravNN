import os
import pickle

import matplotlib.pyplot as plt

from GravNN.Visualization.MapBase import MapBase

map_vis = MapBase()


def plot_1d_curve(df, metric, y_label, size=map_vis.half_page):
    fig, ax = map_vis.newFig(fig_size=map_vis.half_page)
    plt.semilogx(df.index * (df.index + 1), df[metric])
    plt.ylabel(y_label)
    plt.xlabel("Params, $p$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    return fig


def main():
    """
    This generates
    1) Spherical Harmonic RSE median curves Brillouin, LEO, and GEO
    2) Spherical Harmonic RSE 2 sigma curves Brillouin, LEO, and GEO
    3) Spherical Harmonic RSE 2 sigma compliment curves Brillouin, LEO, and GEO
    """

    # TODO: Consider the proper figure size.

    directory = os.path.abspath(".") + "/Plots/OneOff/SH_RSE/"
    os.makedirs(directory, exist_ok=True)

    with open("sh_regress_stats_Brillouin.data", "rb") as f:
        brillouin_df = pickle.load(f)

    # # ! RSE MEDIAN
    # # Brillouin 0 km
    # fig = plot_1d_curve(brillouin_df, 'rse_median', 'Median RSE')
    # map_vis.save(fig, directory + "SH_Median_RSE_Brillouin.pdf")

    # # ! RSE 2 SIGMA MEDIAN
    # # Brillouin 0 km
    # fig = plot_1d_curve(brillouin_df, 'sigma_2_median', 'Median RSE')
    # map_vis.save(fig, directory + "SH_sigma_2_Median_RSE_Brillouin.pdf")

    # # ! RSE 2 SIGMA COMPLIMENT MEDIAN
    # # Brillouin 0 km
    # fig = plot_1d_curve(brillouin_df, 'sigma_2_c_median', 'Median RSE')
    # map_vis.save(fig, directory + "SH_sigma_2_c_Median_RSE_Brillouin.pdf")

    # ! RSE MEDIAN Full Size
    # Brillouin 0 km
    fig, ax = map_vis.newFig(fig_size=map_vis.full_page)
    plt.semilogx(brillouin_df.index, brillouin_df["rse_mean"])
    plt.ylabel("Mean RSE")
    plt.xlabel("Degree, $l$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    map_vis.save(fig, directory + "SH_Mean_RSE_Brillouin_Full.pdf")

    fig, ax = map_vis.newFig(fig_size=map_vis.full_page)
    plt.semilogx(
        brillouin_df.index,
        brillouin_df["rse_mean"],
        label=r"MSE($\mathcal{A}$)",
    )
    plt.semilogx(
        brillouin_df.index,
        brillouin_df["sigma_2_mean"],
        label=r"MSE($\mathcal{F}$)",
    )
    plt.semilogx(
        brillouin_df.index,
        brillouin_df["sigma_2_c_mean"],
        label=r"MSE($\mathcal{C}$)",
    )

    plt.ylabel("Mean RSE")
    plt.xlabel("Degree, $l$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    plt.legend()
    map_vis.save(fig, directory + "SH_Regress_Brill_All.pdf")
    plt.show()


if __name__ == "__main__":
    main()
