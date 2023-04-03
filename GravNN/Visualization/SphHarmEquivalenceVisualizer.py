import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import GravNN
from GravNN.Analysis.SphHarmEquivalenceExperiment import SphHarmEquivalenceExperiment
from GravNN.Networks.Data import DataSet
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sph
from GravNN.Trajectories import RandomDist
from GravNN.Visualization.VisualizationBase import VisualizationBase


class SphHarmEquivalenceVisualizer(VisualizationBase):
    def __init__(self, df):
        self.df = df

    def generate_histogram(self, positions, min_radius):
        ax = plt.gca()
        positions = cart2sph(positions)
        r_mag = positions[:, 0]
        altitude = r_mag - min_radius
        altitude_km = altitude / 1000.0

        ax.hist(
            altitude_km,
            bins=100,
            alpha=0.3,
        )
        ax.set_xlabel("Altitude [km]")
        ax.set_ylabel("Frequency")
        ax.set_xlim([0, None])
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    def generate_altitude_curve(self, df, statistic):
        ids = df["id"].values
        fig_list = []
        labels = []
        linestyles = []
        GravNN_path = os.path.dirname(GravNN.__file__)

        # Generate their altitude rse plot
        for model_id in ids:
            # Load model
            tf.keras.backend.clear_session()
            config, model = load_config_and_model(model_id, df)

            # Gather rse statistics for that model (must run analysis w/ altitude)
            directory = f"{GravNN_path}/Data/Networks/{model_id}/"
            df = pd.read_pickle(directory + "rse_alt.data")

            linestyle = "-"
            if config["PINN_constraint_fcn"][0] == "pinn_00":
                linestyle = "--"

            fig, ax = self.newFig()
            alt_km = df.index / 1000.0
            plt.plot(alt_km, df[statistic], linestyle=linestyle)
            plt.xlabel("Altitude [km]")
            plt.ylabel("Nearest SH Degree")
            ax.yaxis.set_ticks_position("left")

            fig_list.append(fig)
            labels.append(str(config["num_units"][0]))
            linestyles.append(linestyle)
        return fig_list, labels, linestyles

    def plot(
        self,
        df,
        statistic,
        legend=False,
        bounds=None,
    ):
        min_radius = df["radius_min"][0]

        # Generate trajectory histogram
        data = DataSet(self.config)

        main_fig, ax = self.newFig()
        self.generate_histogram(data.raw_data["x_train"], min_radius)
        ax = plt.gca().twinx()

        # Generate the SH equivalent altitude curves individually
        fig_list, labels, linestyles = self.generate_altitude_curve(df, statistic)

        # Take the curves from each figure and put them all on the histogram plot
        handles = []
        colors = ["red", "red", "green", "green", "blue", "blue"]
        for j, fig in enumerate(fig_list):
            cur_fig = plt.figure(fig.number)
            cur_ax = cur_fig.get_axes()[0]

            data = cur_ax.get_lines()[0].get_xydata()
            alt_km = data[:, 0]
            sh_eq = data[:, 1]
            label = f"$N={labels[j]}$" if linestyles[j] == "-" else None

            (line,) = ax.plot(
                alt_km,
                sh_eq,
                label=label,
                linestyle=linestyles[j],
                c=colors[j],
            )
            handles.append(line)
            plt.close(cur_fig)

        ax.set_ylabel("Nearest SH Degree")
        if legend:
            ax.legend(handles=handles, loc="upper right")

        ax.yaxis.set_label_position("left")  # Works
        plt.tick_params(
            axis="y",  # changes apply to the x-axis
            which="minor",  # both major and minor ticks are affected
            left=True,  # ticks along the left edge are off
            right=False,  # ticks along the top edge are off
            labelleft=True,
            labelright=False,
        )  # labels along the left edge are off
        plt.tick_params(
            axis="y",
            which="major",
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
        )
        if bounds is not None:
            ax.set_ylim(bottom=bounds[0], top=bounds[1])


def extract_sub_df(trajectory, df):
    sub_df = df[df["distribution"] == trajectory.__class__]
    sub_df = sub_df[
        (sub_df["num_units"] == 20)
        | (sub_df["num_units"] == 40)
        | (sub_df["num_units"] == 80)
    ].sort_values(by="num_units", ascending=False)
    return sub_df


def main(self):
    trad_df = pd.read_pickle("Data/Dataframes/traditional_nn_df.data")
    pinn_df = pd.read_pickle("Data/Dataframes/pinn_df.data")
    df = pd.concat([trad_df, pinn_df])

    # Compute the equivalent SH for a specified model
    for i in range(df):
        config, model = load_config_and_model(df.id[i], df)

        exp = SphHarmEquivalenceExperiment(model, config, "sh_stats_Brillouin.data")
        exp.run()

    # for all models in the dataframe, visualize the SH equivalent vs altitude
    sub_df = extract_sub_df(RandomDist, df)
    vis = SphHarmEquivalenceVisualizer(sub_df)
    vis.plot(sub_df, statistic="param_rse_mean", legend=True)
    name = "Random"
    self.save(plt.gcf(), f"Generalization/{name}nn_sh_altitude_equivalence.pdf")
