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
    excludes = ['accuracy', 'loss', 'val_loss', 'round_epochs', 'epochs']
    #first = './Hyperparams/Random/070620135028.csv' # Random Min Max 1
    #second = './Hyperparams/Random/070720072710.csv' # Random Min Max 2
    #a = Analyze('./Hyperparams/Random/070720152121.csv') # Random Min Max 3

    #a = Analyze("./Hyperparams/Uniform/070820084805.csv") # Uniform Min Max 1
    #a = Analyze("./Hyperparams/Uniform/070820131402.csv") # Uniform Standard 1
    #a = Analyze("./Hyperparams/Uniform/070820134910.csv") # Uniform Robust 1
    #a = Analyze('./Hyperparams/Uniform/070820172255.csv') # Uniform Min Max 2 Full -- Run overnight, 600 cases, very little variance, wayyy too many parameters
    #a = Analyze('./Hyperparams/Uniform/070920073557.csv') # Uniform Min Max v2.0 -- Wider search space, little variance in results (all near 0.65 accuracy), fewer parameters (~3000)
    a = Analyze('./Hyperparams/Uniform/070920090006.csv') # Uniform Min Max v2.1 -- 1000 points, Narrower search space, prioritizing batch size and smaller first node possibilities. 
    a = Analyze('./Hyperparams/Uniform/070920090006.csv') # Uniform Min Max v2.1.1 -- 10000 points, Same space as above

    # Current goal -- Identify hyperparameter space that shows the most amount of variability among the search space. Right now almost all configurations tested have yielded extremely similar results. It's possible that this is because the diversity of data is quite low. The Uniform Grid really only has 968 * 2 = 1900 free input parameters because the radial component is the same. Its possible that the data just isn't diverse enough to capture more nuanced behavior so the best it can do is 65%. 

    # The new focus is to find what sample distribution provides the correct amount
    df = a.data

    a.plot_corr('val_accuracy', ['acc', 'accuracy', 'loss', 'val_loss', 'round_epochs', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mse'])
    #a.plot_hist('val_accuracy', bins=50)
    a.plot_bars('first_neuron','val_accuracy', 'epochs','lr')
    #a.plot_box('losses', 'val_accuracy', 'epochs')

    plt.show()


if __name__ == '__main__':
    main()