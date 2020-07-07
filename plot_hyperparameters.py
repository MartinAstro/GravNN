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
from Support.transformations import sphere2cart, cart2sph, project_acceleration
from GravityModels.NNSupport.NN_hyperparam import NN_hyperparam

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adadelta, Nadam, Adam, RMSprop
from talos.utils.recover_best_model import recover_best_model
from talos import Analyze
import matplotlib.pyplot as plt
seed(1)

def main():
    excludes = ['shape', 'epochs', 'round_epochs']
    #excludes = []
    a = Analyze('Initial_Search/070620135028.csv')
    a.plot_corr('accuracy', excludes)
    a.plot_hist('accuracy', bins=20)
    df = a.data

    #a.plot_bars(df['lr'], df['val_accuracy'])

    best = a.best_params('accuracy', excludes)
    train_high = a.high('accuracy')
    val_high = a.high("val_accuracy")
    plt.show()


if __name__ == '__main__':
    main()