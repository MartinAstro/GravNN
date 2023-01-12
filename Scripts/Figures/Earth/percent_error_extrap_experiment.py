
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Networks.Configs import get_default_earth_config
from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Planets import Earth
import matplotlib.pyplot as plt
import pandas as pd

def main():

    # Notes: With NN Potential Scaling is [-1,-2]
    # TODO: Subclass ExtrapolationVisualizer to overlay the results 

    # pinn model
    df = pd.read_pickle("Data/Dataframes/earth_percent_error_test.data")

    ################
    # With Percent #
    ################

    model_id = df["id"].values[-3] 
    config, model = load_config_and_model(model_id, df)

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 10000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(extrapolation_exp, x_axis='dist_2_COM', plot_fcn=plt.semilogy, annotate=False)
    vis.fig_size = vis.half_page_default
    vis.plot_interpolation_percent_error()
    plt.gcf().axes[0].set_ylim([1E-2, 1E2])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Cost_with_Percent_Percent.pdf", pad_inches=0.0)
    plt.savefig("Plots/PINNIII/Cost_with_Percent_Percent.png", pad_inches=0.0, dpi=250)
    
    vis.plot_interpolation_rms()
    plt.gcf().axes[0].set_ylim([1E-11, 1E-3])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Cost_with_Percent_RMS.pdf", pad_inches=0.0)
    plt.savefig("Plots/PINNIII/Cost_with_Percent_RMS.png", pad_inches=0.0, dpi=250)

    




    ############
    # With RMS #
    ############

    model_id = df["id"].values[-4] 
    config, model = load_config_and_model(model_id, df)

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 10000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(extrapolation_exp, x_axis='dist_2_COM', plot_fcn=plt.semilogy, annotate=False)
    vis.fig_size = vis.half_page_default

    vis.plot_interpolation_percent_error()
    plt.gcf().axes[0].set_ylim([1E-2, 1E2])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Cost_without_Percent_Percent.pdf", pad_inches=0.0)
    plt.savefig("Plots/PINNIII/Cost_without_Percent_Percent.png", pad_inches=0.0, dpi=250)

    vis.plot_interpolation_rms()
    plt.gcf().axes[0].set_ylim([1E-11, 1E-3])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Cost_without_Percent_RMS.pdf", pad_inches=0.0)
    plt.savefig("Plots/PINNIII/Cost_without_Percent_RMS.png", pad_inches=0.0, dpi=250)


    plt.show()
if __name__ == "__main__":
    main()