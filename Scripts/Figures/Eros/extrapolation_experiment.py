
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

    def plot(self, x, value):
        # compute trend lines
        def get_rolling_lines(data):
            df = pd.DataFrame(data=data, index=None)
            avg = df.rolling(50, 25).mean()
            std = df.rolling(50, 25).std()
            max = df.rolling(10, 10).max()
            return avg, std, max
        
        # sort entries
        avg_line, std_line, max_line = get_rolling_lines(value)
        
        self.newFig()
        plt.scatter(x, value, alpha=0.2, s=2)
        self.plot_fcn(x, avg_line)

        y_std_upper = np.squeeze(avg_line + 1*std_line)
        y_std_lower = np.squeeze(avg_line - 1*std_line)
        plt.fill_between(x, y_std_lower, y_std_upper, color='C0', alpha=0.5)
        self.plot_fcn(x, max_line, color='red')
        
        training_bounds = self.training_bounds / self.radius
        plt.vlines(training_bounds[0], ymin=0, ymax=np.max(value), color='green')
        plt.vlines(training_bounds[1], ymin=0, ymax=np.max(value), color='green')
        plt.vlines(1, ymin=0, ymax=np.max(value), color='grey')
        if self.annotate:
            self.annotate_metrics(value)
        plt.tight_layout()

def main():
    # pinn model
    df = pd.read_pickle("Data/Dataframes/eros_PINN_extrapolation.data")

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
    # vis.plot_interpolation_percent_error()
    vis.plot_extrapolation_percent_error()
    plt.gca().set_xlim([0, 5])
    plt.gca().set_ylim([1E-7, 1E1])
    
    # vis.plot_interpolation_rms()
    # vis.plot_extrapolation_rms()
    # plt.savefig("Plots/PINNIII/U_with_scale_extrap.pdf")





    ############
    # PINN III #
    ############

    model_id = df["id"].values[-2] # without scaling
    config, model = load_config_and_model(model_id, df)

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 5000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizerMod(extrapolation_exp, x_axis='dist_2_COM', plot_fcn=plt.semilogy)
    # vis.plot_interpolation_percent_error()
    vis.plot_extrapolation_percent_error()
    plt.gca().set_xlim([0, 5])
    plt.gca().set_ylim([1E-7, 1E1])
    
    # vis.plot_interpolation_rms()
    # vis.plot_extrapolation_rms()
    # plt.savefig("Plots/PINNIII/U_without_scale_extrap.pdf")

    plt.show()
if __name__ == "__main__":
    main()