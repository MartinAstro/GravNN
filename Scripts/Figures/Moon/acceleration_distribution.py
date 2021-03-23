import matplotlib.pyplot as plt
import numpy as np

from GravNN.Trajectories.FibonacciDist import FibonacciDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.CelestialBodies.Planets import Moon
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Visualization.VisualizationBase import VisualizationBase
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

def main():
    planet = Moon()
    trajectory = RandomDist(planet, [planet.radius, planet.radius+50000.0], 1000000)
    x_unscaled, a_unscaled, u_unscaled = get_sh_data(trajectory, planet.sh_hf_file, deg_removed=2, max_deg=1000)

    vis = VisualizationBase()
    vis.newFig()
    plt.subplot(3,1,1)
    plt.hist(a_unscaled[:,0], bins=1000)
    plt.subplot(3,1,2)
    plt.hist(a_unscaled[:,1], bins=1000)    
    plt.subplot(3,1,3)
    plt.hist(a_unscaled[:,2], bins=1000)


    # min_max_transformer = MinMaxScaler(feature_range=(-1,1))
    # a_scaled = min_max_transformer.fit_transform(a_unscaled)
    # vis.newFig()
    # plt.subplot(3,1,1)
    # plt.hist(a_scaled[:,0], bins=1000)
    # plt.subplot(3,1,2)
    # plt.hist(a_scaled[:,1], bins=1000)    
    # plt.subplot(3,1,3)
    # plt.hist(a_scaled[:,2], bins=1000)

    standard_transformer = QuantileTransformer()
    a_scaled = standard_transformer.fit_transform(a_unscaled)
    vis.newFig()
    plt.subplot(3,1,1)
    plt.hist(a_scaled[:,0], bins=1000)
    plt.subplot(3,1,2)
    plt.hist(a_scaled[:,1], bins=1000)    
    plt.subplot(3,1,3)
    plt.hist(a_scaled[:,2], bins=1000)

    standard_transformer = QuantileTransformer(output_distribution='normal')
    a_scaled = standard_transformer.fit_transform(a_unscaled)
    vis.newFig()
    plt.subplot(3,1,1)
    plt.hist(a_scaled[:,0], bins=1000)
    plt.subplot(3,1,2)
    plt.hist(a_scaled[:,1], bins=1000)    
    plt.subplot(3,1,3)
    plt.hist(a_scaled[:,2], bins=1000)


    plt.show()
main()