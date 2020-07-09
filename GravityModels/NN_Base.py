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

import sigfig
import pickle

import os, sys
sys.path.append(os.path.dirname(__file__) + "/../")
from Support.transformations import cart2sph, sphere2cart, project_acceleration, invert_projection
import inspect
import keras.backend as K

class NN_Base(GravityModelBase):
    def __init__(self, model, preprocessor, test_traj=None):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        super().configure(test_traj)
        return

    def generate_full_file_directory(self):
        self.file_directory +=  self.model.name + "/"
    
    def trainNN(self):
        self.generate_full_file_directory()
        #earlyStop = EarlyStopping(monitor='loss', min_delta=1E-4, patience=self.patience, verbose=1, mode='auto',
                                                #baseline=None, restore_best_weights=False)
        self.saveNN()
        return


    def plotMetrics(self):
        loss = self.fit.history['loss']
        val_loss = self.fit.history['val_loss']
        epochs = range(len(loss))

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if not os.path.exists(self.file_directory):
            os.makedirs(self.file_directory)
        plt.savefig(self.file_directory + 'loss.pdf', bbox_inches='tight')
        return

    def compute_percent_error(self,predicted=None, truth=None):
        if predicted is None and truth is None:
            predicted = self.model.predict(np.array(self.x_test).reshape((len(self.x_test),3)))
            truth = self.y_test
        else:
            predicted = predicted.reshape(len(truth),3)
            truth = truth.reshape(len(truth),3)

        error = np.zeros((4,))
        cumulativeSum = 0.0
        zeros = np.zeros((4,))

        error[0] = np.median(np.abs((predicted[:,0] - truth[:,0])/ truth[:,0]))
        error[1] = np.median(np.abs((predicted[:,1] - truth[:,1])/ truth[:,1]))
        error[2] = np.median(np.abs((predicted[:,2] - truth[:,2])/ truth[:,2]))
        error[3] =  np.median(np.abs(np.linalg.norm(predicted - truth,axis=1)/ np.linalg.norm(truth,axis=1)))
        error *= 100

        print("\n\n\n")
        print("Median Total Error: " + str(error[3]) + "\n")
        print("Component Error")
        print(error[0:3])
        return 

    def predict(self, x):
        y_pred = self.model.predict(np.array(x).reshape((len(x),3)))
        return y_pred.reshape((len(x),1, 3))

    def compute_acc(self, positions=None):
        """Compute the accelerations via NN

        Args:
            positions (np.array): position in cartesian coordinates

        Returns:
            np.array: acceleration array in cartesian coordinates
        """
        if positions is None:
            positions = self.trajectory.positions
        
        positions = cart2sph(positions)
        positions = self.preprocessor.apply_transform(x=positions)[0]
        pred_accelerations_encode = self.predict(positions)
        positions, pred_accelerations_decode = self.preprocessor.invert_transform(x=positions, y=pred_accelerations_encode)

        pred_accelerations_decode = invert_projection(positions, pred_accelerations_decode)

        # If the NN has been asked to compute an acceleration for a trajectory, save the acceleration within NN_dir/Trajectory/Acceleration
        self.accelerations = pred_accelerations_decode

        return pred_accelerations_decode






