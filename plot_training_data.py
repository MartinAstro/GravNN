

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
from Preprocessors.StandardTransform import StandardTransform
from Preprocessors.RobustTransform import RobustTransform
from Preprocessors.QuantileTransform import QuantileTransform
from Preprocessors.MaxAbsTransform import MaxAbsTransform


from Support.transformations import sphere2cart, cart2sph, project_acceleration

from GravityModels.GravityModelBase import GravityModelBase
from GravityModels.NN_Base import NN_Base

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta
import copy
def plots(preprocessor, x_decode, y_decode):
    preprocessor.split(x_decode, y_decode)
    preprocessor.fit()
    x_encode, x_test, y_encode, y_test = preprocessor.apply_transform()

    plt.figure()
    plt.subplot(221)
    plt.title(preprocessor.__class__.__name__)
    plt.hist(x_decode, bins=100)
    plt.ylabel("Pre Position")
    plt.subplot(222)
    plt.hist(y_decode,bins=100)
    plt.ylabel("Pre Acceleration")
  
    plt.subplot(223)
    plt.title("Post Transform")
    plt.hist(x_encode, bins=100)
    plt.ylabel("Post Position")
    plt.subplot(224)
    plt.ylabel("Post Acceleration")
    plt.hist(y_encode,bins=100)
    plt.show()

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

    plots(MinMaxTransform(val_range=[0,1]), pos_sphere, acc_proj)
    plots(StandardTransform(), pos_sphere, acc_proj)
    plots(RobustTransform(), pos_sphere, acc_proj)
    plots(QuantileTransform(), pos_sphere, acc_proj)
    plots(MaxAbsTransform(), pos_sphere, acc_proj)




if __name__ == "__main__":
    main()