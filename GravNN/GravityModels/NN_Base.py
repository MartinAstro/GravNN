import inspect
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.Support.transformations import (cart2sph, invert_projection,
                                            project_acceleration, sphere2cart)
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l1
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


class NN_Base(GravityModelBase):
    def __init__(self, model, preprocessor, test_traj=None):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        super().configure(test_traj)
        return

    def generate_full_file_directory(self):
        self.file_directory +=  self.model.name + "/"
    
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






