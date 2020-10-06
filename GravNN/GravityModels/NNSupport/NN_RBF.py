import inspect
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import sigfig
import tensorflow.keras.backend as K
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.GravityModels.NNSupport.kmeans_initializer import InitCentersKMeans
from GravNN.GravityModels.NNSupport.rbflayer import RBFLayer
from GravNN.Support.transformations import (cart2sph, invert_projection,
                                            project_acceleration, sphere2cart)
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.utils import class_weight
from talos.model.normalizers import lr_normalizer
from talos.utils import hidden_layers
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l1, l1_l2, l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


def NN_RBF(x_train, y_train, x_val, y_val, params, verbose=0, save_location=None, validation_split=None, lr_norm=False):
    X = np.hstack((x_train, y_train))
    rbflayer = RBFLayer(output_dim=params['first_unit'],
                                        initializer=InitCentersKMeans(X),
                                        betas=params['beta'],
                                        input_shape=(x_train.shape[1],))
    model = Sequential()
    model.add(rbflayer)
    for i in range(params['hidden_layers']-1):
        rbflayer = RBFLayer(output_dim=params['first_unit'],
                                            betas=params['beta'],
                                            )
        model.add(rbflayer)
        model.add(Dropout(params['dropout']))

    model.add(Dense(units=y_train.shape[1], activation='linear'))

    if lr_norm:
        optimizer = params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer']))
    else:
        optimizer = params['optimizer'](lr=params['lr'])
    model.compile(loss=params['losses'],
                                optimizer=optimizer,
                                metrics=['mse', 'mae']
                                ) # accuracy doesn't tell you anything
                                #https://datascience.stackexchange.com/questions/48346/multi-output-regression-problem-with-keras
    # dot_img_file = '/Users/johnmartin/Desktop/model_1.png'
    # plot_model(model, to_file=dot_img_file, show_shapes=True)

    #earlyStop = EarlyStopping(monitor='loss', min_delta=1E-4, patience=patience, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
    if validation_split is not None:

        history = model.fit(x_train, y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=verbose,
                    validation_split=validation_split)
                    #validation_data=(x_val, y_val))
                    #callbacks=[talos.utils.live()])
                    #callbacks=[earlyStop])
    else:
        history = model.fit(x_train, y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=verbose,
                    validation_data=(x_val, y_val))
                    #callbacks=[talos.utils.live()])
                    #callbacks=[earlyStop])

    if save_location is not None:
        os.makedirs(save_location,exist_ok=True)
        model_json = model.to_json()
        with open(save_location+"model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(save_location + "model.h5")
        with open(save_location + "history.data", 'wb') as f:
            pickle.dump(history.history, f)
    
    return history, model
