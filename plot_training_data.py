

from Visualization.Grid import Grid
from Visualization.MapVisualization import MapVisualization
from GravityModels.SphericalHarmonics import SphericalHarmonics
from CelestialBodies.Planets import Earth
from Trajectories.DHGridDist import DHGridDist
from GravityModels.NN_Base import NN_Base

import pyshtools
import matplotlib.pyplot as plt
import numpy as np

from Trajectories.UniformDist import UniformDist
from Trajectories.RandomDist import RandomDist

from Trajectories.DHGridDist import DHGridDist
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics
from Preprocessors.MinMaxTransform import MinMaxTransform
from Support.transformations import sphere2cart, cart2sph, project_acceleration

from GravityModels.GravityModelBase import GravityModelBase
from GravityModels.NN_Base import NN_Base

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta
import copy

def main():
    density_deg = 175
    max_deg = 1000
    planet = Earth()
    sh_file = planet.sh_hf_file

    # Loop through different NN configurations
    nn_list = []
    #points = [1000, 10000]#, 100000]
    points = 10000
    trajectory = UniformDist(planet, planet.radius, points)
    #trajectory = RandomDist(planet, [planet.radius, planet.radius*1.1], point_count)
    training_gravity_model = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    training_gravity_model.load(override=False) 

    # Subtract off C20 from training data
    training_gravity_model_C20 = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory)
    training_gravity_model_C20.load(override=False) 
    training_gravity_model.accelerations -= training_gravity_model_C20.accelerations

    pos_sphere = cart2sph(trajectory.positions)
    acc_proj = project_acceleration(pos_sphere, training_gravity_model.accelerations)

    plt.figure()
    plt.subplot(211)
    plt.title("Pre Transform")
    plt.hist(pos_sphere)
    plt.subplot(212)
    plt.hist(acc_proj,bins=100)
    plt.show()

    preprocessor = MinMaxTransform(val_range=[-1,1])
    preprocessor.split(pos_sphere, acc_proj)
    preprocessor.fit()
    x_train, x_test, y_train, y_test = preprocessor.apply_transform()

    plt.figure()
    plt.subplot(211)
    plt.title("Post Transform")
    plt.hist(x_train)
    plt.subplot(212)
    plt.hist(y_train,bins=100)
    plt.show()

if __name__ == "__main__":
    main()