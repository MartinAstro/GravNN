import os, sys
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["RUNFILES_DIR"] = "/Users/johnmartin/Library/Python/3.7/share/plaidml"# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path
# os.environ["PLAIDML_NATIVE_PATH"] = "/Users/johnmartin/Library/Python/3.7/lib/libplaidml.dylib" # libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path

import keras
import pickle
import talos
import numpy as np
from numpy.random import seed

from Trajectories.UniformDist import UniformDist
from Trajectories.DHGridDist import DHGridDist
from Trajectories.RandomDist import RandomDist
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics
from Preprocessors.MinMaxTransform import MinMaxTransform
from Support.transformations import sphere2cart, cart2sph, project_acceleration
from GravityModels.NNSupport.NN_hyperparam import NN_hyperparam

from GravityModels.NN_Base import NN_Base
from GravityModels.NNSupport.NN_DeepLayers import *
from GravityModels.NNSupport.NN_MultiLayers import *
from GravityModels.NNSupport.NN_SingleLayers import *

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adadelta, Nadam, Adam, RMSprop
from talos import Analyze, Reporting, Evaluate, Predict, Restore
seed(1)

def main():
    planet = Earth()
    point_count = 1000
    #trajectory = UniformDist(planet, planet.radius, 10000)
    trajectory = RandomDist(planet, [planet.radius, planet.radius*1.1], point_count)

    gravityModel = SphericalHarmonics(planet.sh_file, degree=None, trajectory=trajectory)
    gravityModel.load() 

    pos_sphere = cart2sph(trajectory.positions)
    acc_proj = project_acceleration(pos_sphere, gravityModel.accelerations)

    preprocessor = MinMaxTransform()
    preprocessor.percentTest = 0.01
    preprocessor.split(pos_sphere, acc_proj)
    preprocessor.fit()
    x_train, x_test, y_train, y_test = preprocessor.apply_transform()

    p = {'first_neuron':[128, 64, 16],
            'hidden_layers':[1, 3],
            'batch_size': [5, 15],
            'epochs': [100, 200, 300],
            'dropout': [0.0, 0.2],
            'lr' : [0.1, 0.15, 0.2],
            'kernel_initializer': ['normal'],
            'optimizer': [Nadam, Adadelta],
            'shapes' : ['brick'], 
            'losses': ['mean_squared_error', 'mean_absolute_error'],
            'activation':['relu', 'elu', 'tanh']
           }

    t = talos.Scan(x=x_train,
                              y=y_train,
                              model=NN_hyperparam,
                              params=p,
                              experiment_name='Initial_Search',
                              fraction_limit=0.02)





if __name__ == '__main__':
    main()