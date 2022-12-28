
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


def plot(data, planet=None, log=False):
    if planet is not None:
        R = planet.radius
        unit = "[R]"
    else:
        R = 1
        unit = '[m]'

    r = np.linalg.norm(data.raw_data['x_train'], axis=1) / R
    u = np.linalg.norm(data.raw_data['u_train'], axis=1)
    plt.figure()
    plt.scatter(r, u)
    max_u = np.max(u)
    plt.scatter(r, max_u/r**1, label='inv1', s=2)
    plt.scatter(r, max_u/r**2, label='inv2', s=2)
    plt.scatter(r, max_u/r**3, label='inv3', s=2)
    plt.scatter(r, max_u/r**4, label='inv4', s=2)
    plt.scatter(r, max_u/r**5, label='inv5', s=2)
    plt.scatter(r, max_u/r**6, label='inv6', s=2)
    plt.xlabel(f"Radius {unit}")
    plt.ylabel("Potential")

    plt.legend()
    if log:
        plt.gca().set_yscale("log")

    plt.figure()
    plt.scatter(r, u*r**3)
    plt.ylabel("Potential Scaled")



def main():
    # spherical harmonic model 
    planet = Earth()
    max_degree = 10
    degree_removed = -1 
    
    #l = 0, power = 3
    #l = 1, power = 3
    #l = 2, power = 4
    #l = 3, power = 5

    config = get_data_config(max_degree, degree_removed, max_radius=planet.radius*10) 
    data = DataSet(data_config=config)
    plot(data, planet, log=True)
    
    plt.show()
if __name__ == "__main__":
    main()