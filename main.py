import os, sys
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["RUNFILES_DIR"] = "/Users/johnmartin/Library/Python/3.7/share/plaidml"# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path
os.environ["PLAIDML_NATIVE_PATH"] = "/Users/johnmartin/Library/Python/3.7/lib/libplaidml.dylib" # libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path

import keras
import pickle
import numpy as np
from numpy.random import seed

from States import States
from Trajectories.UniformDist import UniformDist
from CelestialBodies.Planets import Earth
from AccelerationAlgs.SphericalHarmonics import SphericalHarmonics
from Preprocessors.MinMaxTransform import MinMaxTransform

from NN.NN_Base import NN_Base
from NN.NN_DeepLayers import *
from NN.NN_MultiLayers import *
from NN.NN_SingleLayers import *

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta
seed(1)

def main():
    planet = Earth()
    trajectory = UniformDist(planet, planet.geometry.radius + 50000, 10000)
    accResults = SphericalHarmonics(trajectory, degree=10)
    preprocessor = MinMaxTransform()
    preprocessor.percentTest = 0.1

    nn = NN_Base(trajectory, accResults, preprocessor)
    nn.epochs = 1000
    nn.batch_size = 100
    nn.lr = 5E-2
    nn.opt = Adadelta()
    #nn.opt = SGD(lr=nn.lr) 
    #nn.loss = "mean_absolute_percentage_error"

    nn.model_func = Single_128_LReLU
    nn.forceNewNN = False
    nn.trainNN()

    # Plotting Routines
    nn.plotMetrics()  # Plot keras generated validation/test accuracy/error
    nn.computePercentError() # Plot Relative Error

if __name__ == '__main__':
    main()