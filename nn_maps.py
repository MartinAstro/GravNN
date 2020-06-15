import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["RUNFILES_DIR"] = "/Users/johnmartin/Library/Python/3.7/share/plaidml"# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path
os.environ["PLAIDML_NATIVE_PATH"] = "/Users/johnmartin/Library/Python/3.7/lib/libplaidml.dylib" # libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path

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
from Trajectories.DHGridDist import DHGridDist
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics
from Preprocessors.MinMaxTransform import MinMaxTransform
from support.transformations import sphere2cart, cart2sph, project_acceleration

from GravityModels.NN_Base import NN_Base
from GravityModels.NNSupport.NN_DeepLayers import *
from GravityModels.NNSupport.NN_MultiLayers import *
from GravityModels.NNSupport.NN_SingleLayers import *

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta
from Visualization.MapVizFunctions import * 

def main():
    planet = Earth()
    radius = planet.radius + 50000
    trajectory = UniformDist(planet, radius, 10000)
    training_gravity_model = SphericalHarmonics(planet.sh_file, degree=None, trajectory=trajectory)
    training_gravity_model.load() 

    preprocessor = MinMaxTransform()
    preprocessor.percentTest = 0.1
        
    nn = NN_Base(trajectory, training_gravity_model, preprocessor)
    nn.epochs = 1000
    nn.batch_size = 1
    nn.lr = 5E-2
    nn.opt = Adadelta()
    #nn.opt = SGD(lr=nn.lr) 
    #nn.loss = "mean_absolute_percentage_error"

    nn.model_func = Single_128_LReLU
    nn.forceNewNN = False
    nn.trainNN()




    # Main 
    map_grid = DHGridDist(planet, planet.radius + 50000, degree=175)
    sh_C20_gravityModel = SphericalHarmonics(planet.sh_file, degree=2, trajectory=map_grid)
    sh_all_gravityModel = SphericalHarmonics(planet.sh_file, degree=None, trajectory=map_grid)

    map_viz = MapVisualization()

    true_grid = Grid(gravityModel=sh_all_gravityModel)
    sh_20_grid = Grid(gravityModel=sh_C20_gravityModel)

    nn.trajectory = map_grid
    nn_grid = Grid(gravityModel=nn)

    high_fidelity_maps(nn_grid, sh_20_grid)
    percent_error_maps(true_grid, nn_grid, sh_20_grid, vlim=500)
    # component_error([5, 10, 20, 50, 100, 150], radius, planet.sh_file, true_grid, C20_grid) # Takes a long time

    plt.show()




if __name__ == "__main__":
    main()
