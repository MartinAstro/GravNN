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

import sigfig
import pickle

import keras.backend as K

class NN_Base:
    def __init__(self, trajectory, accResults, preprocessor):
        self.x_train, self.x_test, self.y_train, self.y_test = preprocessor.split(trajectory.positions, accResults.accelerations)
        self.x_train, self.x_test, self.y_train, self.y_test = preprocessor.apply_transform()

        self.epochs = None
        self.batch_size = None
        self.lr = None
        self.loss = 'mean_squared_error' # 'mean_absolute_error', 'mean_squared_logarithmic_error',
        self.opt = None
        self.model_func = None
        self.forceNewNN = False

        self.file_directory = accResults.file_directory 

        self.verbose = 1
        return

    def set_NN_path(self):
        self.file_directory += \
                 self.model_func.__name__ + "/" + \
                 "Epochs_" + str(self.epochs) + \
                 "_opt_" + str(self.opt.__class__.__name__) + \
                 "_bs_" + str(self.batch_size) + \
                 "_lr_" + str(self.lr) + \
                 "_loss_" + str(self.loss)  + "/"

    
    def load(self):
        try:
            self.model = load_model(self.file_directory + "model.h5")
            self.fit = pickle.load(open(self.file_directory + "fit.data", 'rb'))
        except:
            print("Unable to load " + self.file_directory + "model.h5")

    def save(self):
        try:
            self.model.save(self.file_directory + "model.h5")
            pickle.dump(self.fit, open(self.file_directory + "fit.data", 'wb'))
        except:
            print("Couldn't save NN")
            pass

    def trainNN(self):
        self.set_NN_path()
        if not self.forceNewNN and os.path.exists(self.file_directory):
            self.load()
        else:
            if not os.path.exists(self.file_directory):
                os.makedirs(self.file_directory)
            self.model = self.model_func(self.x_train, self.y_train)
            self.model.compile(
                            loss=self.loss,
                            optimizer=self.opt,
                            metrics=['mae'])
            earlyStop = EarlyStopping(monitor='loss', min_delta=1E-4, patience=10, verbose=1, mode='auto',
                                                    baseline=None, restore_best_weights=False)
            self.fit = self.model.fit(self.x_train, self.y_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        verbose=self.verbose,
                        validation_split=0.2,
                        callbacks=[earlyStop])

            self.save()
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
        plt.savefig(self.file_directory + 'loss.png')
        return

    def computePercentError(self,):
        error = np.zeros((len(self.y_test[0]),))
        cumulativeSum = 0.0
        zeros = np.zeros((3,))
        # Compute the percent error in each coefficient averaged across all test data
        for i in range(len(self.x_test)):
            y_pred = self.model.predict(np.array([self.x_test[i]]))
            for k in range(len(y_pred[0])):
                if np.abs(self.y_test[i][k]) < 1E-12:
                    zeros[k] += 1
                else:
                    error[k] += np.abs(y_pred[0][k] - self.y_test[i][k]) / (np.abs(self.y_test[i][k]))

        error[0] /= (len(self.x_test) - zeros[0])
        error[1] /= (len(self.x_test) - zeros[1])
        error[2] /= (len(self.x_test) - zeros[2])
        error *= 100

        cumulativeSum = np.sum(error)
        cumulativeSum /= len(self.x_test[0])

        print("\n\n\n")
        print("Average Total Error: " + str(cumulativeSum) + "\n")
        print("Component Error")
        print(error)