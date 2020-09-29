import inspect
import os
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import sys
sys.path.append(os.path.dirname(__file__) + "/../")

import matplotlib.pyplot as plt
import numpy as np
import sigfig
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.utils import class_weight
from talos.model.normalizers import lr_normalizer
import tensorflow as tf
from talos.utils import hidden_layers
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l1, l1_l2, l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from GravityModels.GravityModelBase import GravityModelBase
from Support.transformations import (cart2sph, invert_projection,
                                     project_acceleration, sphere2cart)
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase



def NN_Conv(x_train, y_train, x_val, y_val, params, verbose=0, save_location=None, validation_split=None, lr_norm=False):

    #r, theta, phi acceleration maps
    r_field_input = keras.Input(shape=(28,28,1),  name='r_map')
    x = layers.Conv2D(128, 3, activation="relu")(r_field_input)

    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64,3, activation="relu")(x)

    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(32,3, activation="relu")(x)

    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(16,3, activation="relu")(x)


    theta_field_input = keras.Input(shape=(28,28,1),  name='theta_map')
    x = layers.Conv2D(16, 3, activation="relu")(r_field_input)
    x = layers.Conv2D(32,3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)

    phi_field_input = keras.Input(shape=(28,28,1),  name='phi_map')
    x = layers.Conv2D(16, 3, activation="relu")(r_field_input)
    x = layers.Conv2D(32,3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)

    pos_input = keras.Input(shape=(3,1), name="pos_input")

    x = layers.concatenate([r_field_input, theta_field_input, phi_field_input, pos_input])

    acc_pred = layers.Dense(3, name="acceleration")(x)

    model = keras.Model(inputs=[r_field_input, theta_field_input, phi_field_input, pos_input],
                                            outputs=[acc_pred])
                                            
    if lr_norm:
        optimizer = params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer']))
    else:
        optimizer = params['optimizer'](lr=params['lr'])
    model.compile(loss=params['losses'],
                                optimizer=optimizer,
                                metrics=['mse', 'mae']
                                ) # accuracy doesn't tell you anything
                                #https://datascience.stackexchange.com/questions/48346/multi-output-regression-problem-with-keras
    dot_img_file = '/Users/johnmartin/Desktop/conv_network.png'
    plot_model(model, to_file=dot_img_file, show_shapes=True)

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


def NN_Conv_Simple(total, pos, acc, params, verbose=0, save_location=None, validation_split=None, lr_norm=False):

    #r, theta, phi acceleration maps
    pos_input = keras.Input(shape=(3,1), name="pos_input")
    y = layers.Dense(8, activation='relu')(pos_input)
    y = layers.Dense(16, activation='relu')(y)
    y = layers.Dense(32, activation='relu')(y)
    pos_condense = layers.Flatten()(y)


    field_input = keras.Input(shape=(len(total),len(total[0]),1),  name='total_map')
    # x = layers.Conv2D(128, 3, activation="relu")(field_input)
    # x = layers.MaxPooling2D((2,2))(x)
    # x = layers.Conv2D(64,3, activation="relu")(field_input)#(x)
    # x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(32,3, activation="relu")(field_input)#(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(16,3, activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(8,3, activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(4,3, activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    field_condense = layers.Flatten()(x)

    x = layers.concatenate([field_condense, pos_condense])
    x = layers.Dense(128, name="intermediate", activation='relu')(x)
    x = layers.Dense(128,  activation='relu')(x)

    acc_pred = layers.Dense(3, name="acceleration", activation='linear')(x)
    model = keras.Model(inputs=[field_input, pos_input],
                                            outputs=[acc_pred])
                                            
    model.compile(loss=params['losses'],
                                optimizer=Adadelta(),
                                metrics=['mse', 'mae']
                                ) 
    dot_img_file = '/Users/johnmartin/Desktop/conv_network.png'
    plot_model(model, to_file=dot_img_file, show_shapes=True)

    images = np.full((len(pos),np.shape(total)[0],np.shape(total)[1]), total)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs", 
                                                                                                    write_images=True,
                                                                                                    profile_batch=1,
                                                                                                    histogram_freq=1)

    history = model.fit(
                    {"total_map": images, "pos_input": pos},
                    {"acceleration": acc},
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=verbose,
                    validation_split=0.1,
                    callbacks=[tensorboard_callback])
    #earlyStop = EarlyStopping(monitor='loss', min_delta=1E-4, patience=patience, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

    if save_location is not None:
        os.makedirs(save_location,exist_ok=True)
        model_json = model.to_json()
        with open(save_location+"model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(save_location + "model.h5")
        with open(save_location + "history.data", 'wb') as f:
            pickle.dump(history.history, f)
    
    return history, model
