import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Analysis.CompactnessExperiment import CompactnessExperiment
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Visualization.VisualizationBase import VisualizationBase


class CompactnessVisualizer(VisualizationBase):
    def __init__(self, exp, log=True, **kwargs):
        self.exp = exp
        self.plot_fcn = plt.semilogx if log else plt.plot
        super().__init__(**kwargs)

    def sh_pareto_curve(
        self,
        sigma,
        metric,
        label="MRSE",
    ):
        sh_df = self.exp.sh_df
        params = sh_df.index * (sh_df.index + 1)
        sh_rse = sh_df["rse_" + metric]
        sh_sigma = sh_df[f"sigma_{sigma}_{metric}"]
        sh_sigma_c = sh_df[f"sigma_{sigma}_c_{metric}"]

        self.plot_fcn(params, sh_rse, label=label + r"($\mathcal{A}$)")
        self.plot_fcn(params, sh_sigma, label=label + r"($\mathcal{F}$)")
        self.plot_fcn(params, sh_sigma_c, label=label + r"($\mathcal{C}$)")

        ax = plt.gca()
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        plt.ylabel("MRSE [m/s$^2$]")
        plt.xlabel("Parameters")

    def nn_pareto_curve(
        self,
        sigma,
        metric,
        linestyle=None,
        marker=None,
    ):
        nn_df = self.exp.nn_df
        nn_df.sort_values(by=["params"], inplace=True)

        plt.gca().set_prop_cycle(None)

        params = nn_df.params
        df_rse = nn_df[f"rse_{metric}"]
        df_sigma = nn_df[f"sigma_{sigma}_{metric}"]
        df_sigma_c = nn_df[f"sigma_{sigma}_c_{metric}"]

        self.plot_fcn(params, df_rse, linestyle=linestyle, marker=marker)
        self.plot_fcn(params, df_sigma, linestyle=linestyle, marker=marker)
        self.plot_fcn(params, df_sigma_c, linestyle=linestyle, marker=marker)
        plt.legend()

    def plot(self, sigma, metric="mean"):
        self.newFig()
        self.sh_pareto_curve(sigma, metric)
        self.nn_pareto_curve(
            sigma,
            metric,
            linestyle="-",
            marker="o",
        )


if __name__ == "__main__":
    # sh_df = pd.read_pickle("Data/Dataframes/sh_stats_Brillouin.data")
    # sh_df = pd.read_pickle("Data/Dataframes/sh_stats_Brillouin_deg_n1.data")
    # nn_df = pd.read_pickle("Data/Dataframes/pinn_df.data")
    nn_df = pd.read_pickle("Data/Dataframes/earth_PINN_III_FF_040423.data")

    planet = Earth()
    exp = CompactnessExperiment(nn_df, 2)
    exp.run(planet, planet.radius, points=50000)

    vis = CompactnessVisualizer(exp)
    vis.plot(sigma=2)
    plt.show()
