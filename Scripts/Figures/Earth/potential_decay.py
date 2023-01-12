
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
from GravNN.Networks.Data import DataSet
from GravNN.Visualization.VisualizationBase import VisualizationBase

plt.rc('text', usetex=True)

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


def get_data_config(max_degree, deg_removed, max_radius):
    config = get_default_earth_config()
    config.update({
        "radius_max" : [max_radius],
        "N_dist": [10000],
        "N_train": [9500],
        "N_val": [500],
        "max_deg" : [max_degree],
        "deg_removed": [deg_removed],
        "dummy_transformer": [DummyScaler()],
    })
    return config


def plot(data, planet, log=False, deg_removed=None):

    R = planet.radius
    r = np.linalg.norm(data.raw_data['x_train'], axis=1) / R
    u_train = data.raw_data['u_train'].squeeze()
    u = u_train / np.max(np.abs(u_train))
    vis = VisualizationBase()
    plt.rc('font', size=7)
    vis.fig_size = vis.half_page_default
    vis.newFig()
    plt.scatter(r, u, alpha=0.5, s=2, label='$U$')
    max_u = np.max(u)

    plt.xlabel(f"Planet Radii from Surface [-]")
    plt.ylabel("N.D. Potential $\delta U$ [-]")



    if log:
        plt.gca().set_yscale("log")

    power = 0
    if deg_removed is not None:
        if deg_removed == -1:
            power = 1
        elif np.any(deg_removed == [0,1]):
            power = 3
        else:
            power = deg_removed + 2 

    plt.scatter(r, max_u/r**power, alpha=0.5, s=2, label='$\\frac{1}{r^p}$')
    plt.legend()
    plt.tight_layout()

    vis.newFig()
    plt.scatter(r, u*r**power, alpha=0.5, s=2)
    plt.ylabel("Scaled N.D. Potential\n $U_{NN} = \delta U * r^p$ [-]")
    plt.xlabel(f"Planet Radii from Surface [-]")
    plt.tight_layout()


def main():
    # spherical harmonic model 
    planet = Earth()
    max_degree = 1000
    degree_removed = 2 
    
    #l = 0, power = 3
    #l = 1, power = 3
    #l = 2, power = 4
    #l = 3, power = 5

    config = get_data_config(max_degree, degree_removed, max_radius=planet.radius*5) 
    data = DataSet(data_config=config)
    plot(data, planet=planet, log=False, deg_removed=degree_removed)
    
    plt.figure(1)
    plt.savefig("Plots/PINNIII/Potential_NoScale.pdf")
    plt.savefig("Plots/PINNIII/Potential_NoScale.png", dpi=300)

    plt.figure(2)
    plt.savefig("Plots/PINNIII/Potential_Scale.pdf")
    plt.savefig("Plots/PINNIII/Potential_Scale.png", dpi=300)
    

    plt.show()
if __name__ == "__main__":
    main()