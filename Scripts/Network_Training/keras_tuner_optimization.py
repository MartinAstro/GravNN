import os
import pickle
import sys

from GravNN.build.PinesAlgorithm import PinesAlgorithm
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.NN_Base import NN_Base
from GravNN.GravityModels.NNSupport.NN_keras_tuner import NN_keras_tuner
from GravNN.GravityModels.NNSupport.SupportFunc import *

from GravNN.Preprocessors.MaxAbsTransform import MaxAbsTransform
from GravNN.Preprocessors.MinMaxStandardTransform import \
    MinMaxStandardTransform
from GravNN.Preprocessors.MinMaxTransform import MinMaxTransform
from GravNN.Preprocessors.RobustTransform import RobustTransform
from GravNN.Preprocessors.StandardTransform import StandardTransform
from GravNN.Support.transformations import (cart2sph,
                                            check_fix_radial_precision_errors,
                                            project_acceleration, sphere2cart)
from GravNN.GravityModels.NNSupport.SupportFunc import plot_metrics
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from keras.utils.layer_utils import count_params
from numpy.random import seed
from sklearn.model_selection import train_test_split
from talos import Analyze, Deploy, Evaluate, Predict, Reporting, Restore, Scan
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from kerastuner.tuners import RandomSearch

earlyStop =EarlyStopping(
    monitor='loss', 
    min_delta=1E-4, 
    patience=5, 
    verbose=1, 
    mode='auto', 
    baseline=None, 
    restore_best_weights=False)

tensorboard = TensorBoard(
    log_dir="logs",
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    update_freq="epoch",
    profile_batch=2,
    embeddings_freq=0,
    embeddings_metadata=None,
)

lrPlateau = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=5,
    verbose=1,
    mode="auto",
    min_delta=0.01,
    cooldown=0,
    min_lr=0,
    )

seed(1)

def main():
    planet = Earth()
    preprocessor = MinMaxStandardTransform()
    point_count = 259200 
    #point_count = 180*360

    # trajectory = UniformDist(planet, planet.radius, point_count)
    trajectory = RandomDist(planet, [planet.radius, planet.radius+5000.0], point_count)
    #trajectory = RandomDist(planet, [planet.radius+330.0*1000-2500 , planet.radius + 330.0*1000+2500], point_count) #LEO

    experiment_dir = generate_experiment_dir(trajectory, preprocessor)

    sh_file = planet.sh_hf_file
    max_deg = 1000

    gravityModelMap = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    gravityModelMap.load() 
    gravityModelMapC20 = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory)
    gravityModelMapC20.load() 
    gravityModelMap.accelerations -= gravityModelMapC20.accelerations

    pos_sphere = cart2sph(trajectory.positions)
    pos_sphere = check_fix_radial_precision_errors(pos_sphere)
    acc_proj = project_acceleration(pos_sphere, gravityModelMap.accelerations)
    
    preprocessor.percentTest = 0.3
    preprocessor.split(pos_sphere, acc_proj)
    preprocessor.fit()
    x_train, x_val, y_train, y_val = preprocessor.apply_transform()

    tuner = RandomSearch(NN_keras_tuner,
                                                objective='val_loss',
                                                max_trials=5,
                                                executions_per_trial=1,
                                                directory='Scripts/Hyperparams/keras_tuner/',
                                                project_name='example')

    tuner.search_space_summary()
    tuner.search(x_train, y_train, 
                            epochs=30, 
                            batch_size=64,
                            validation_data=(x_val, y_val))

    tuner.results_summary()

    # Get best model and retrain
    histories = []
    best_hps = tuner.get_best_hyperparameters(num_trials=2)
    for hps in best_hps:
        model = tuner.hypermodel.build(hps)
        history = model.fit(x_train, y_train, 
                                epochs=30, 
                                batch_size=64,
                                callbacks=[lrPlateau],
                                validation_data=(x_val, y_val))
        histories.append(history)

    plot_metrics(histories)
    plt.show()




if __name__ == '__main__':
    main()
