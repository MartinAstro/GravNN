import os, sys
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["RUNFILES_DIR"] = "/Users/johnmartin/Library/Python/3.7/share/plaidml"# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path
# os.environ["PLAIDML_NATIVE_PATH"] = "/Users/johnmartin/Library/Python/3.7/lib/libplaidml.dylib" # libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path

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
from Preprocessors.RobustTransform import RobustTransform

from Support.transformations import sphere2cart, cart2sph, project_acceleration
from GravityModels.NNSupport.NN_hyperparam import NN_hyperparam
from GravityModels.NN_Base import NN_Base

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adadelta, Nadam, Adam, RMSprop
from talos import Analyze, Reporting, Evaluate, Predict, Restore, Scan
import matplotlib.pyplot as plt

seed(1)

def plot_metrics(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()    
    return

def main():
    planet = Earth()
    point_count = 1000
    trajectory = UniformDist(planet, planet.radius, point_count)
    #trajectory = RandomDist(planet, [planet.radius, planet.radius*1.1], point_count)

    gravityModelMap = SphericalHarmonics(planet.sh_file, degree=None, trajectory=trajectory)
    gravityModelMap.load() 
    gravityModelMapC20 = SphericalHarmonics(planet.sh_file, degree=2, trajectory=trajectory)
    gravityModelMapC20.load() 
    gravityModelMap.accelerations -= gravityModelMapC20.accelerations

    pos_sphere = cart2sph(trajectory.positions)
    acc_proj = project_acceleration(pos_sphere, gravityModelMap.accelerations)

    preprocessor = MinMaxTransform()
    # a = Analyze("./Hyperparams/Initial_Search/Uniform/070820084805.csv") # Uniform Min Max 1
    a = Analyze('./Hyperparams/Uniform/070920073557.csv') # Uniform Min Max v2.0

    # preprocessor = RobustTransform()
    # a = Analyze("./Hyperparams/Initial_Search/Uniform/070820134910.csv") # Uniform Robust 1
    preprocessor.percentTest = 0.0
    preprocessor.split(pos_sphere, acc_proj)
    preprocessor.fit()
    x_train, y_train = preprocessor.apply_transform()

   
    df = a.data
    run = df[df['val_accuracy'] == df['val_accuracy'].max()]
    params = run.iloc[0].to_dict()
    #params['optimizer'] = Nadam
    params['optimizer'] = SGD
    params['losses'] = 'mean_absolute_error'# 'mean_squared_error'


    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)
    hist, model = NN_hyperparam(x_train, y_train, x_val, y_val, params, verbose=1)
    plot_metrics(hist)
    plt.show()
    return



if __name__ == '__main__':
    main()