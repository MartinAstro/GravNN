
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Networks.Configs import get_default_earth_config
from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Planets import Earth
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ExtrapolationVisualizerMod(ExtrapolationVisualizer):
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        plt.rc('text', usetex=True)


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
        
        if kwargs.get('newFig', True):
            self.newFig()
        plt.scatter(x, value, alpha=0.2, s=2)
        self.plot_fcn(x, avg_line)

        
        training_bounds = self.training_bounds / self.radius
        plt.vlines(training_bounds[0], ymin=0, ymax=np.max(value), color='green')
        plt.vlines(training_bounds[1], ymin=0, ymax=np.max(value), color='green')
        plt.vlines(1, ymin=0, ymax=np.max(value), color='grey')
        plt.tight_layout()

def main():
    # pinn model
    df = pd.read_pickle("Data/Dataframes/eros_PINN_III_extrapolation_v3.data")

    ###########
    # PINN II #
    ###########

    model_id = df["id"].values[-1] # with scaling
    config, model = load_config_and_model(model_id, df)

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 5000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizerMod(extrapolation_exp, x_axis='dist_2_COM', plot_fcn=plt.semilogy)
    vis.fig_size = vis.full_page_silver
    vis.plot_extrapolation_percent_error()
    




    ############
    # PINN III #
    ############

    model_id = df["id"].values[-2] # without scaling
    config, model = load_config_and_model(model_id, df)
    config['fuse_models'] = [False]

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 5000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizerMod(extrapolation_exp, x_axis='dist_2_COM', plot_fcn=plt.semilogy)
    vis.fig_size = vis.full_page_silver

    vis.plot_extrapolation_percent_error(newFig=False)
    plt.gca().set_xlim([0, 10])
    plt.gca().set_ylim([1E-4, 1E1])

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4)]

    plt.gca().legend(custom_lines, ['PINN-GM-II', 'PINN-GM-III'])


    plt.savefig("Plots/PINNIII/Eros_extrapolation_IIvIII.pdf")

    plt.show()
if __name__ == "__main__":
    main()