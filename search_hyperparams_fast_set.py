import os
import pickle
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
import talos
from keras.utils.layer_utils import count_params
from numpy.random import seed
from talos import Analyze, Deploy, Evaluate, Predict, Restore, Scan
from talos.utils.recover_best_model import recover_best_model
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l1, l2

from CelestialBodies.Planets import Earth
from GravityModels.NN_Base import NN_Base
from GravityModels.NNSupport.NN_hyperparam import NN_hyperparam
from GravityModels.SphericalHarmonics import SphericalHarmonics
from Preprocessors.MinMaxTransform import MinMaxTransform
from Preprocessors.RobustTransform import RobustTransform
from Preprocessors.StandardTransform import StandardTransform
from Support.transformations import (cart2sph,
                                     check_fix_radial_precision_errors,
                                     project_acceleration, sphere2cart)
from Trajectories.DHGridDist import DHGridDist
from Trajectories.RandomDist import RandomDist
from Trajectories.UniformDist import UniformDist
from Visualization.Grid import Grid
from Visualization.MapVisualization import MapVisualization

seed(1)

"""
The goal of this search is to find the NN that is able to capture the most accuracy for the fewest data. A 50x50 field needs a total of 2550 unique points (likely much larger actually), but until I have a way to regress I don't know. Might look at EGM96 to see how much data they used. 
"""
planet = Earth()
point_count = 2592 #10368
sh_file = planet.sh_hf_file
max_deg = 1000
trajectory = UniformDist(planet, planet.radius, point_count)
#trajectory = RandomDist(planet, [planet.radius, planet.radius*1.1], point_count)
preprocessor = MinMaxTransform()
#preprocessor = StandardTransform()
#preprocessor = RobustTransform()
experiment_dir = 'Hyperparams/Uniform/MinMax/100'
model_name = "hyper_" + experiment_dir.split('/')[1:].replace("/","_")
model_name += "_v10_0"
frac = 0.25

# Uniform 10000 case Min Max
p = {'first_unit':[8, 128, 256],
        'first_neuron' : [8, 16, 32],
        'hidden_layers':[0, 1, 2],
        'batch_size': [20],
        'epochs': [100, 300],
        'dropout': [0.0, 0.2, 0.4],
        'lr' : [0.2, 0.4],
        'kernel_initializer': [None],#['normal', 'uniform'],
        'kernel_regularizer': [None],# ['l2', 'l1', None], 
        'optimizer': [Nadam],#, Adadelta],
        'shapes' : ['brick'], 
        'losses': ['mean_squared_error'],
        'activation':['relu']#, 'elu', 'tanh']
        }

gravityModel = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
gravityModel.load() 

gravityModelC20 = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory)
gravityModelC20.load()

pos_sphere = cart2sph(trajectory.positions)
acc_proj = project_acceleration(pos_sphere, gravityModel.accelerations)
acc_projC20 = project_acceleration(pos_sphere, gravityModelC20.accelerations)
acc_proj = acc_proj - acc_projC20

preprocessor.percentTest = 0.3
preprocessor.split(pos_sphere, acc_proj)
preprocessor.fit()
x_train, x_val, y_train, y_val = preprocessor.apply_transform()

t = talos.Scan(x=x_train,
                            y=y_train,
                            x_val=x_val,
                            y_val=y_val,
                            model=NN_hyperparam,
                            params=p,
                            experiment_name=experiment_dir,
                            fraction_limit=frac, 
                            minimize_loss=True,
                            reduction_metric='val_loss')

d = Deploy(t, model_name, 'val_loss', asc=True)

e = Evaluate(t)
eval = e.evaluate(x_val, 
                                y_val, 
                                model_id=None, 
                                folds=5, 
                                shuffle=True, 
                                metric='val_loss', 
                                task='continuous',
                                asc=True,
                                print_out=True)
