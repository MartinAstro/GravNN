
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Networks.Configs import get_default_earth_config
from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Planets import Earth
import matplotlib.pyplot as plt
import pandas as pd

class SphericalHarmonicWoPointMass(SphericalHarmonics):
    def __init__(self, sh_info, max_deg, remove_deg):
        self.high_fidelity_model = SphericalHarmonics(sh_info, max_deg)
        self.low_fidelity_model = SphericalHarmonics(sh_info, remove_deg)
    
    def compute_acceleration(self, positions=None):
        high_acc = self.high_fidelity_model.compute_acceleration(positions)
        low_acc = self.low_fidelity_model.compute_acceleration(positions)
        return high_acc - low_acc

    def compute_potential(self, positions=None):
        high_pot = self.high_fidelity_model.compute_potential(positions)
        low_pot = self.low_fidelity_model.compute_potential(positions)
        return high_pot - low_pot


def get_sh_model(max_degree, deg_removed):
    planet = Earth()
    model = SphericalHarmonicWoPointMass(planet.EGM2008, max_degree, deg_removed)

    # update the true "training data"
    config = get_default_earth_config()
    config.update({
        # "radius_max" : [Earth().radius*5],
        # "radius_max" : [Earth().radius + 420000],
        "radius_min" : [Earth().radius*1],
        "radius_max" : [Earth().radius*15],
        "N_dist": [10000],
        "N_train": [9500],
        "N_val": [500],
        "deg_removed": [deg_removed],
        "dummy_transformer": [DummyScaler()],
    })
    return config, model

def main():
    # spherical harmonic model 
    # config, model = get_sh_model(max_degree=33, deg_removed=2)



    # pinn model
    df = pd.read_pickle("Data/Dataframes/high_altitude_behavior.data")

    ################
    # With Scaling #
    ################

    model_id = df["id"].values[-1] # with scaling
    config, model = load_config_and_model(model_id, df)

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 10000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(extrapolation_exp, x_axis='dist_2_COM', plot_fcn=plt.semilogy, annotate=False)
    vis.fig_size = vis.half_page_default
    vis.plot_extrapolation_rms(plot_std=False)
    plt.gca().set_ylim([1E-17, 1E-3])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/U_with_scale_extrap.pdf", pad_inches=0.0)





    ###################
    # Without Scaling #
    ###################

    model_id = df["id"].values[-2] # without scaling
    config, model = load_config_and_model(model_id, df)

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 10000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(extrapolation_exp, x_axis='dist_2_COM', plot_fcn=plt.semilogy, annotate=False)
    vis.fig_size = vis.half_page_default
    vis.plot_extrapolation_rms(plot_std=False)
    plt.gca().set_ylim([1E-17, 1E-3])
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/U_without_scale_extrap.pdf", pad_inches=0.0)

    plt.show()
if __name__ == "__main__":
    main()