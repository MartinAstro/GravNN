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
    def __setattr__(self, name, value):
        if name is "trajectory" and issubclass(value.__class__, TrajectoryBase):
            self.file_directory += value.trajectory_name + "/"
        super(NN_Base, self).__setattr__(name, value)


    def __init__(self, train_trajectory, train_gravity_model, preprocessor):
        super().__init__()
        self.file_directory = train_gravity_model.file_directory

        # preprocess the data
        pos_sphere = cart2sph(train_trajectory.positions)
        acc_proj = project_acceleration(pos_sphere, train_gravity_model.accelerations)

        self.preprocessor = preprocessor
        self.preprocessor.split(pos_sphere, acc_proj)
        self.preprocessor.fit()
        self.x_train, self.x_test, self.y_train, self.y_test = self.preprocessor.apply_transform()

        self.epochs = None
        self.batch_size = None
        self.lr = None
        self.loss = 'mean_squared_error' # 'mean_absolute_error', 'mean_squared_logarithmic_error',
        self.opt = None
        self.patience = None
        self.model_func = None
        self.forceNewNN = False

        self.nn_verbose = 1
        return

    def generate_full_file_directory(self):
        self.file_directory += \
                 self.model_func.__name__ + "/" + \
                 "Epochs_" + str(self.epochs) + \
                 "_opt_" + str(self.opt.__class__.__name__) + \
                 "_bs_" + str(self.batch_size) + \
                 "_lr_" + str(self.lr) + \
                 "_loss_" + str(self.loss)  +  \
                 "_patience_" + str(self.patience) + "/"

    
    def importNN(self):
        try:
            self.model = load_model(self.file_directory + "model.h5")
            self.fit = pickle.load(open(self.file_directory + "fit.data", 'rb'))
        except:
            print("Unable to load " + self.file_directory + "model.h5")

    def saveNN(self):
        try:
            self.model.save(self.file_directory + "model.h5")
            pickle.dump(self.fit, open(self.file_directory + "fit.data", 'wb'))
        except:
            print("Couldn't save NN")
            pass

    def trainNN(self):
        self.generate_full_file_directory()
        if not self.forceNewNN and os.path.exists(self.file_directory):
            self.importNN()
        else:
            if not os.path.exists(self.file_directory):
                os.makedirs(self.file_directory)
            self.model = self.model_func(self.x_train, self.y_train)
            self.model.compile(
                            loss=self.loss,
                            optimizer=self.opt,
                            metrics=['mae'])
            earlyStop = EarlyStopping(monitor='loss', min_delta=1E-4, patience=self.patience, verbose=1, mode='auto',
                                                    baseline=None, restore_best_weights=False)
            self.fit = self.model.fit(self.x_train, self.y_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        verbose=self.nn_verbose,
                        validation_split=0.2,
                        callbacks=[earlyStop])

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

        # for i in range(len(truth)):
        #     for k in range(len(truth[0])):
        #         if np.abs(truth[i][k]) < 1E-12:
        #             zeros[k] += 1
        #         else:
        #             error[k] += np.abs((predicted[i][k] - truth[i][k]) /(truth[i][k]))

        # error[0] /= (len(truth) - zeros[0])
        # error[1] /= (len(truth) - zeros[1])
        # error[2] /= (len(truth) - zeros[2])
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






