import os, sys
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["RUNFILES_DIR"] = "/Users/johnmartin/Library/Python/3.7/share/plaidml"# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path
os.environ["PLAIDML_NATIVE_PATH"] = "/Users/johnmartin/Library/Python/3.7/lib/libplaidml.dylib" # libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path

import keras
import pickle
import numpy as np
from numpy.random import seed

from Trajectories.UniformDist import UniformDist
from Trajectories.DHGridDist import DHGridDist
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics
from Preprocessors.MinMaxTransform import MinMaxTransform
from Support.transformations import sphere2cart, cart2sph, project_acceleration

from GravityModels.NN_Base import NN_Base
from GravityModels.NNSupport.NN_DeepLayers import *
from GravityModels.NNSupport.NN_MultiLayers import *
from GravityModels.NNSupport.NN_SingleLayers import *

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta
seed(1)

def main():
    planet = Earth()
    trajectory = UniformDist(planet, planet.radius + 50000, 10000)
    gravityModel = SphericalHarmonics(planet.sh_file, degree=None, trajectory=trajectory)
    gravityModel.load() 

    preprocessor = MinMaxTransform()
    preprocessor.percentTest = 0.1
        
    nn = NN_Base(trajectory, gravityModel, preprocessor)
    nn.epochs = 1000
    nn.batch_size = 1
    nn.lr = 5E-2
    nn.opt = Adadelta()
    #nn.opt = SGD(lr=nn.lr) 
    #nn.loss = "mean_absolute_percentage_error"

    nn.model_func = Single_128_LReLU
    nn.forceNewNN = False
    nn.trainNN()

    # Plotting Routines
    nn.plotMetrics()  # Plot keras generated validation/test accuracy/error
    nn.compute_percent_error() # Plot Relative Error



    # At this point the NN has been saved under trajectories + acceleration algorithm + NN directory
    # To now test the performance of the particular NN, need to generate a new data set and load it through the NN
    map_grid = DHGridDist(planet, planet.radius + 50000, degree=175)
    gravityModel = SphericalHarmonics(planet.sh_file, degree=None, trajectory=map_grid)
    gravityModel.load(override=False)

    pred_accelerations = nn.compute_acc(map_grid.positions)
    true_accelerations = gravityModel.accelerations

    nn.compute_percent_error(pred_accelerations, true_accelerations)


if __name__ == '__main__':
    main()