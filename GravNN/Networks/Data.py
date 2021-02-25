import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.GravityModels.PointMass import PointMass


from GravNN.Support.transformations import project_acceleration, cart2sph

def training_validation_split(X, Y, Z, N_train, N_val):

    X, Y, Z = shuffle(X, Y, Z, random_state=42)

    X_train = X[:N_train]
    Y_train = Y[:N_train]
    Z_train = Z[:N_train]

    X_val = X[N_train:N_train+N_val]
    Y_val = Y[N_train:N_train+N_val]
    Z_val = Z[N_train:N_train+N_val]

    return X_train, Y_train, Z_train, X_val, Y_val, Z_val

def generate_dataset(x, y, batch_size):
    x = x.astype('float32')
    y = y.astype('float32')
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    #dataset = dataset.shuffle(1000, seed=1234)
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()

    #Why Cache is Impt: https://stackoverflow.com/questions/48240573/why-is-tensorflows-tf-data-dataset-shuffle-so-slow
    return dataset