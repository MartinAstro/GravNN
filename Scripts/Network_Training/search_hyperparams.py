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
from GravNN.Preprocessors.MinMaxTransform import MinMaxTransform
from GravNN.Preprocessors.MinMaxStandardTransform import MinMaxStandardTransform
from GravNN.Preprocessors.RobustTransform import RobustTransform
from GravNN.Preprocessors.StandardTransform import StandardTransform
from GravNN.Support.transformations import (cart2sph,
                                     check_fix_radial_precision_errors,
                                     project_acceleration, sphere2cart)
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization

seed(1)

planet = Earth()
point_count = 259200 # 0.5 Deg
#point_count = 64800 # 1 Deg
#point_count = 10368 #2.5 Deg
#point_count = 2592  # 5.0 Deg

sh_file = planet.sh_hf_file
max_deg = 1000
#trajectory = UniformDist(planet, planet.radius, point_count)
trajectory = RandomDist(planet, [planet.radius, planet.radius+5000], point_count) #5 kilometer band
preprocessor = MinMaxStandardTransform()
#preprocessor = MinMaxTransform()

case = 1
experiment_dir = 'Hyperparams/' +  \
                                trajectory.__class__.__name__ + "/" +    \
                                preprocessor.__class__.__name__ + "/" + \
                                str(point_count)
model_name = "hyper_" + experiment_dir[12:].replace("/","_")
if case is not None:
        experiment_dir += "/case_" + str(case) + "/"
        model_name += "_case_" + str(case)
os.makedirs(experiment_dir, exist_ok=True)

print(model_name)
frac = 1.0

if case == 1:
        p = {
                'first_unit':[13],
                'first_neuron' : [0],
                'hidden_layers':[0],
        }
if case == 2:
        p = {
                'first_unit':[141],
                'first_neuron' : [0],
                'hidden_layers':[0],
        }
if case == 3:
        p = {
                'first_unit':[1428],
                'first_neuron' : [0],
                'hidden_layers':[0],
        }

if case == 4:
        p = {
                'first_unit':[5],
                'first_neuron' : [5],
                'hidden_layers':[2],
        }
if case == 5:
        p = {
                'first_unit':[20],
                'first_neuron' : [20],
                'hidden_layers':[2],
        }
if case == 6:
        p = {
                'first_unit':[68],
                'first_neuron' : [68],
                'hidden_layers':[2],
        }

# ORIGINAL
# p.update( {
#         'batch_size': [50],
#         'epochs': [100],
#         'dropout': [0.0, 0.3],
#         'lr' : [0.15, 0.25],
#         'kernel_initializer': ['normal', 'uniform'],
#         'kernel_regularizer': ['l2', 'l1', None], 
#         'optimizer': [Nadam, Adadelta],
#         'shapes' : ['brick'], 
#         'losses': ['mean_squared_error'],
#         'activation':['relu', 'elu']
#         }
# )



if case == 1:
        p.update({
                'optimizer':[Adadelta, Nadam],
                'kernel_initializer': ['glorot_normal'],
        })
if case == 2:
        p.update({
                'optimizer':[Adadelta],
                'kernel_initializer': ['glorot_uniform'],
        })
if case == 3:
        p.update({
                'optimizer':[Nadam],
                'kernel_initializer': ['glorot_uniform', 'glorot_normal'],
        })
if case == 4:
        p.update({
                'optimizer':[Adadelta],
                'kernel_initializer': ['glorot_uniform', 'glorot_normal'],
        })
if case == 5:
        p.update({
                'optimizer':[Adadelta],
                'kernel_initializer': ['glorot_uniform', 'glorot_normal'],
        })
if case == 6:
        p.update({
                'optimizer':[Adadelta, Nadam],
                'kernel_initializer': ['glorot_normal'],
        })


if case == 1:
        p.update({
                'activation':['elu'],
                'lr' : [0.1, 0.2, 0.3],
        })
if case == 2:
        p.update({
                'activation':['relu'],
                'lr' : [0.1, 0.2, 0.3],
        })
if case == 3:
        p.update({
                'activation':['relu'],
                'lr' : [0.1, 0.2, 0.3],
        })
if case == 4:
        p.update({
                'activation':['relu'],
                'lr' : [0.1, 0.2, 0.3],
        })
if case == 5:
        p.update({
                'activation':['relu'],
                'lr' : [0.25, 0.3, 0.35],
        })
if case == 6:
        p.update({
                'activation':['relu'],
                'lr' : [0.15, 0.2, 0.25],
        })

p.update( {
        'batch_size': [10],
        'epochs': [30],
        'dropout': [0.0],
        'kernel_regularizer': [None], 
        'shapes' : ['brick'], 
        'losses': ['mean_squared_error'],
        }
)

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
                            #fraction_limit=frac, 
                            minimize_loss=True,
                            reduction_metric='val_loss')

#d = Deploy(t, model_name, 'val_loss', asc=True)

# e = Evaluate(t)
# eval = e.evaluate(x_val, 
#                                 y_val, 
#                                 model_id=None, 
#                                 folds=5, 
#                                 shuffle=True, 
#                                 metric='val_loss', 
#                                 task='continuous',
#                                 asc=True,
#                                 print_out=True)
