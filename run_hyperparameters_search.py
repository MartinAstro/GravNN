import os, sys
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
from Preprocessors.StandardTransform import StandardTransform
from Preprocessors.RobustTransform import RobustTransform

from Support.transformations import sphere2cart, cart2sph, project_acceleration, check_fix_radial_precision_errors
from GravityModels.NNSupport.NN_hyperparam import NN_hyperparam

from GravityModels.NN_Base import NN_Base
from Visualization.Grid import Grid
from Visualization.MapVisualization import MapVisualization
from tensorflow.keras.optimizers import SGD, Adadelta, Nadam, Adam, RMSprop
from tensorflow.keras.regularizers import l2, l1
seed(1)

from talos.utils.recover_best_model import recover_best_model
from talos import Analyze, Evaluate, Scan, Predict, Deploy, Restore
import matplotlib.pyplot as plt
from plot_nn_maps import print_nn_params
from keras.utils.layer_utils import count_params

planet = Earth()
point_count = 10000
trajectory = UniformDist(planet, planet.radius, point_count)
#trajectory = RandomDist(planet, [planet.radius, planet.radius*1.1], point_count)

gravityModel = SphericalHarmonics(planet.sh_file, degree=None, trajectory=trajectory)
gravityModel.load() 

gravityModelC20 = SphericalHarmonics(planet.sh_file, degree=2, trajectory=trajectory)
gravityModelC20.load()

pos_sphere = cart2sph(trajectory.positions)
acc_proj = project_acceleration(pos_sphere, gravityModel.accelerations)
acc_projC20 = project_acceleration(pos_sphere, gravityModelC20.accelerations)
acc_proj = acc_proj - acc_projC20

preprocessor = MinMaxTransform()
#preprocessor = StandardTransform()
#preprocessor = RobustTransform()
preprocessor.percentTest = 0.0
preprocessor.split(pos_sphere, acc_proj)
preprocessor.fit()
x_train, y_train = preprocessor.apply_transform()

p = {'first_unit':[512, 32, 8],
        'first_neuron' : [8, 16],
        'hidden_layers':[0, 1],
        'batch_size': [1,10, 20],
        'epochs': [50, 150, 250],
        'dropout': [0.0, 0.4],
        'lr' : [0.05, 0.1, 0.2],
        'kernel_initializer': ['normal', 'uniform'],
        'kernel_regularizer': ['l2', 'l1', None], 
        'optimizer': [Nadam, Adadelta],
        'shapes' : ['brick'], 
        'losses': ['mean_squared_error'],
        'activation':['relu', 'elu', 'tanh']
        }

t = talos.Scan(x=x_train,
                            y=y_train,
                            model=NN_hyperparam,
                            params=p,
                            experiment_name='Hyperparams/Uniform',
                            fraction_limit=0.01)
                            
d = Deploy(t, "hyper_uniform_minmax_v2_1_2", 'val_loss', asc=True)

