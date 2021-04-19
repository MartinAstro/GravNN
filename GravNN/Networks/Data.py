import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.GravityModels.SphericalHarmonics import (SphericalHarmonics,
                                                     get_sh_data)
from GravNN.Networks.Constraints import *
from GravNN.Support.transformations import project_acceleration, cart2sph, sphere2cart
def print_stats(data, name):
    print("(" + name + ") " + "Max: %.4f, Min: %.4f, Range: %.4f" % (np.max(data), np.min(data), np.max(data) - np.min(data)))

def standardize_output(y_hat, config):
    u = np.zeros((len(y_hat), 1))
    a = np.zeros((len(y_hat), 3))
    laplace = np.zeros((len(y_hat), 1))
    curl = np.zeros((len(y_hat), 3))

    if config['PINN_constraint_fcn'][0] == no_pinn:
        a = y_hat
    elif config['PINN_constraint_fcn'][0] == pinn_A:
        a = y_hat

    elif config['PINN_constraint_fcn'][0] == pinn_AL:
        a = y_hat[:,0:3]
        laplace = y_hat[:,3]
    elif config['PINN_constraint_fcn'][0] == pinn_ALC:
        a = y_hat[:,0:3]
        laplace = y_hat[:,3]
        curl = y_hat[:,4:]

    elif config['PINN_constraint_fcn'][0] == pinn_AP:
        u = y_hat[:,0]
        a = y_hat[:,1:4]
    elif config['PINN_constraint_fcn'][0] == pinn_APL:
        u = y_hat[:,0]
        a = y_hat[:,1:4]
        laplace = y_hat[:,4]
    elif config['PINN_constraint_fcn'][0] == pinn_APLC:
        u = y_hat[:,0]
        a = y_hat[:,1:4]
        laplace = y_hat[:,4]
        curl = y_hat[:,5:]
    
    return u, a, laplace, curl

def pull_data(config):
    # TODO: Trajectories should take keyword arguments so the inputs dont have to be standard, just pass in config.
    trajectory = config['distribution'][0](config['planet'][0], [config['radius_min'][0], config['radius_max'][0]], config['N_dist'][0], **config)
    if "Planet" in config['planet'][0].__module__:
        get_analytic_data_fcn = get_sh_data
    else:
        get_analytic_data_fcn = get_poly_data
    x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(trajectory, config['grav_file'][0], **config)


    if config['basis'][0] == 'spherical':
        x_unscaled = cart2sph(x_unscaled)     
        a_unscaled = project_acceleration(x_unscaled, a_unscaled)
        x_unscaled[:,1:3] = np.deg2rad(x_unscaled[:,1:3])

    x_train, a_train, u_train, x_val, a_val, u_val = training_validation_split(x_unscaled, 
                                                                                a_unscaled, 
                                                                                u_unscaled, 
                                                                                config['N_train'][0], 
                                                                                config['N_val'][0])

    print_stats(x_train, "Position")
    print_stats(a_train, "Acceleration")
    print_stats(u_train, "Potential")

    a_train = a_train + config['acc_noise'][0]*np.std(a_train)*np.random.randn(a_train.shape[0], a_train.shape[1])


    # Preprocessing
    x_transformer = config['x_transformer'][0]
    a_transformer = config['a_transformer'][0]
    u_transformer = config['u_transformer'][0]

    # # Scale (a,u) with a_transformer
    # x_train = x_transformer.fit_transform(x_train)
    # a_train = a_transformer.fit_transform(a_train)
    # u_train = a_transformer.transform(np.repeat(u_train,3,axis=1))[:,0].reshape((-1,1))

    # x_val = x_transformer.transform(x_val)
    # a_val = a_transformer.transform(a_val)
    # u_val = a_transformer.transform(np.repeat(u_val,3,axis=1))[:,0].reshape((-1,1))
    # u_transformer = a_transformer

    # Scale (a,u) with u_transformer
    x_train = x_transformer.fit_transform(x_train)
    u_train = u_transformer.fit_transform(np.repeat(u_train,3,axis=1))[:,0].reshape((-1,1))
    a_train = u_transformer.transform(a_train)

    x_val = x_transformer.transform(x_val)
    a_val = u_transformer.transform(a_val)
    u_val = u_transformer.transform(np.repeat(u_val,3,axis=1))[:,0].reshape((-1,1))
    a_transformer = u_transformer

    print_stats(a_train, "Acceleration")
    print_stats(u_train, "Potential")

    laplace_train = np.zeros((np.shape(u_train)))
    laplace_val = np.zeros((np.shape(u_val)))

    curl_train = np.zeros((np.shape(a_train)))
    curl_val = np.zeros((np.shape(a_val)))


    # Decide to train with potential or not
    # TODO: Modify which variables are added to the state to speed up training and minimize memory footprint. 
    if config['PINN_constraint_fcn'][0] == no_pinn:
        y_train = np.hstack([a_train]) 
        y_val = np.hstack([a_val]) 

    # Just use acceleration
    elif config['PINN_constraint_fcn'][0] == pinn_A:
        y_train = np.hstack([a_train]) 
        y_val = np.hstack([a_val]) 

    # Accelerations + 2nd order constraints
    elif config['PINN_constraint_fcn'][0] == pinn_AL:
        y_train = np.hstack([a_train, laplace_train]) 
        y_val = np.hstack([a_val, laplace_val]) 
    elif config['PINN_constraint_fcn'][0] == pinn_ALC:
        y_train = np.hstack([a_train, laplace_train, curl_train]) 
        y_val = np.hstack([a_val, laplace_val, curl_val]) 

    # Accelerations + Potential + 2nd order constraints
    elif config['PINN_constraint_fcn'][0] == pinn_AP:
        y_train = np.hstack([u_train, a_train]) 
        y_val = np.hstack([u_val, a_val]) 
    elif config['PINN_constraint_fcn'][0] == pinn_APL:
        y_train = np.hstack([u_train, a_train, laplace_train]) 
        y_val = np.hstack([u_val, a_val, laplace_val]) 
    elif config['PINN_constraint_fcn'][0] == pinn_APLC:
        y_train = np.hstack([u_train, a_train, laplace_train, curl_train]) 
        y_val = np.hstack([u_val, a_val, laplace_val, curl_val]) 

    else:
        exit("No PINN Constraint Selected!")

    dataset = generate_dataset(x_train, y_train, config['batch_size'][0], dtype=config['dtype'][0])
    val_dataset = generate_dataset(x_val, y_val, config['batch_size'][0], dtype=config['dtype'][0])

    transformers = {"x" : x_transformer,
                    "a" : a_transformer,
                    "u" : u_transformer}

    return dataset, val_dataset, transformers


def training_validation_split(X, Y, Z, N_train, N_val):

    X, Y, Z = shuffle(X, Y, Z, random_state=42)

    X_train = X[:N_train]
    Y_train = Y[:N_train]
    Z_train = Z[:N_train]

    X_val = X[N_train:N_train+N_val]
    Y_val = Y[N_train:N_train+N_val]
    Z_val = Z[N_train:N_train+N_val]

    return X_train, Y_train, Z_train, X_val, Y_val, Z_val

def generate_dataset(x, y, batch_size, dtype=None):
    if dtype is None:
        x = x.astype('float32')
        y = y.astype('float32')
    else:
        if dtype == tf.float32:
            x = x.astype('float32')
            y = y.astype('float32')
        elif dtype == tf.float64:
            x = x.astype('float64')
            y = y.astype('float64')
        else:
            exit("No dtype specified")
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    #dataset = dataset.shuffle(1000, seed=1234)
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()

    #Why Cache is Impt: https://stackoverflow.com/questions/48240573/why-is-tensorflows-tf-data-dataset-shuffle-so-slow
    return dataset