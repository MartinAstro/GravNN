import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import *
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
from keras.models import load_model
from GravityModels.GravityModelBase import GravityModelBase
from Trajectories.TrajectoryBase import TrajectoryBase
from talos.utils import hidden_layers

import sigfig
import pickle
import talos

import os, sys
sys.path.append(os.path.dirname(__file__) + "/../")
from Support.transformations import cart2sph, sphere2cart, project_acceleration, invert_projection
import inspect
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LeakyReLU
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import Ones
from talos.model.normalizers import lr_normalizer
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU


def NN_hyperparam(x_train, y_train, x_val, y_val, params, verbose=0):
    model = Sequential()
    model.add(Dense(units=params['first_unit'], 
                                      input_dim=x_train.shape[1],
                                      activation=params['activation'],
                                      kernel_initializer=params['kernel_initializer'],
                                      kernel_regularizer=params['kernel_regularizer']))

    model.add(Dropout(params['dropout']))
    hidden_layers(model, params, y_train.shape[1])
    model.add(Dense(units=y_train.shape[1], activation='linear'))

    model.compile(loss=params['losses'],
                                optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                                metrics=['mse'])

    #earlyStop = EarlyStopping(monitor='loss', min_delta=1E-4, patience=patience, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
    history = model.fit(x_train, y_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=verbose,
                validation_data=(x_val, y_val))
                #callbacks=[talos.utils.live()])
                #callbacks=[earlyStop])

    return history, model