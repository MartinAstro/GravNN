
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

plt.rc('text', usetex=True)

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
    a_train = data.raw_data['a_train'].squeeze()
    a = a_train / np.max(np.abs(a_train))
    a = a_train 
    min_a = np.min(a)
    max_a = np.max(a)
    plt.figure()
    plt.subplot(1,3,1)
    plt.scatter(r, a[:,0], alpha=0.5, s=2, label='$a_1$')
    plt.ylim([min_a, max_a])
    plt.grid()
    plt.subplot(1,3,2)
    plt.scatter(r, a[:,1], alpha=0.5, s=2, label='$a_2$')
    plt.ylim([min_a, max_a])
    plt.grid()
    plt.subplot(1,3,3)
    plt.scatter(r, a[:,2], alpha=0.5, s=2, label='$a_3$')
    plt.ylim([min_a, max_a])
    plt.grid()
    max_u = np.max(a)

    from GravNN.Support.transformations_tf import cart2sph, compute_projection_matrix
    x_sph = cart2sph(data.raw_data['x_train'].squeeze())
    BN = compute_projection_matrix(x_sph)

    # x_B = tf.matmul(BN, tf.reshape(x, (-1,3,1))) 
    # this will give ~[1, 1E-8, 1E-8]

    a_sph = np.reshape(a, (-1, 3,1))
    a = np.matmul(BN, a_sph).squeeze()
    plt.figure()
    plt.subplot(1,3,1)
    plt.scatter(r, a[:,0], alpha=0.5, s=2, label='$a_1$')
    plt.ylim([min_a, max_a])
    plt.grid()
    plt.subplot(1,3,2)
    plt.scatter(r, a[:,1], alpha=0.5, s=2, label='$a_2$')
    plt.ylim([min_a, max_a])
    plt.grid()
    plt.subplot(1,3,3)
    plt.scatter(r, a[:,2], alpha=0.5, s=2, label='$a_3$')
    plt.ylim([min_a, max_a])
    plt.grid()

def main():
    # spherical harmonic model 
    planet = Earth()
    max_degree = 1000
    degree_removed = 5
    
    config = get_data_config(max_degree, degree_removed, max_radius=planet.radius*5) 
    data = DataSet(data_config=config)
    plot(data, planet=planet, log=False, deg_removed=degree_removed)
    
    plt.show()
if __name__ == "__main__":
    main()