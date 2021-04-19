import os
import pickle
import sys
import time
import warnings
import multiprocessing

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from GravNN.Support.Grid import Grid
from GravNN.Networks import utils
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize, fmin_l_bfgs_b

np.random.seed(1234)

class CustomModel(tf.keras.Model):
    # Initialize the class
    def __init__(self, config, network):
        super(CustomModel, self).__init__()
        self.config = config
        self.network = network
        self.eval = config['PINN_constraint_fcn'][0]
        self.mixed_precision = tf.constant(self.config['mixed_precision'][0], dtype=tf.bool)
        self.variable_cast = config['dtype'][0]

    def call(self, x, training=None):
        return self.eval(self.network, x, training)
    
    #@tf.function(experimental_compile=True)
    def train_step(self, data):
        x, y = data 
        with tf.GradientTape() as tape:
            y_hat = self(x, training=True)
            loss = self.compiled_loss(y, y_hat)
            loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(gradients)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}
        
    #@tf.function(experimental_compile=True)
    def test_step(self, data):
        x, y = data
        y_hat = self(x, training=True)
        loss = self.compiled_loss(y, y_hat)
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

    # https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
    def optimize(self, dataset):
        import tensorflow_probability as tfp

        class History:
            def __init__(self):
                self.history = []
        
        self.history = History()
        def function_factory(model, loss, train_x, train_y):
            """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

            Args:
                model [in]: an instance of `tf.keras.Model` or its subclasses.
                loss [in]: a function with signature loss_value = loss(pred_y, true_y).
                train_x [in]: the input part of training data.
                train_y [in]: the output part of training data.

            Returns:
                A function that has a signature of:
                    loss_value, gradients = f(model_parameters).
            """

            # obtain the shapes of all trainable parameters in the model
            shapes = tf.shape_n(model.trainable_variables)
            n_tensors = len(shapes)

            # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
            # prepare required information first
            count = 0
            idx = [] # stitch indices
            part = [] # partition indices

            for i, shape in enumerate(shapes):
                n = np.product(shape)
                idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
                part.extend([i]*n)
                count += n

            part = tf.constant(part)

            @tf.function#(experimental_compile=True)
            def assign_new_model_parameters(params_1d):
                """A function updating the model's parameters with a 1D tf.Tensor.

                Args:
                    params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
                """

                params = tf.dynamic_partition(params_1d, part, n_tensors)
                for i, (shape, param) in enumerate(zip(shapes, params)):
                    model.trainable_variables[i].assign(tf.reshape(param, shape))

            # now create a function that will be returned by this factory
            @tf.function#(experimental_compile=True)
            def f(params_1d):
                """A function that can be used by tfp.optimizer.lbfgs_minimize.

                This function is created by function_factory.

                Args:
                params_1d [in]: a 1D tf.Tensor.

                Returns:
                    A scalar loss and the gradients w.r.t. the `params_1d`.
                """

                # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
                with tf.GradientTape() as tape:
                    # update the parameters in the model
                    assign_new_model_parameters(params_1d)
                    # calculate the loss
                    U_dummy = tf.zeros_like(train_x[:,0:1])

                    #U_dummy = tf.zeros((tf.divide(tf.size(train_x),tf.constant(3)),1))
                    pred_y = model(train_x, training=True)
                    loss_value = loss(pred_y, train_y)

                # calculate gradients and convert to 1D tf.Tensor
                grads = tape.gradient(loss_value, model.trainable_variables)
                grads = tf.dynamic_stitch(idx, grads)

                # print out iteration & loss
                f.iter.assign_add(1)
                tf.print("Iter:", f.iter, "loss:", loss_value)

                # store loss value so we can retrieve later
                tf.py_function(f.history.append, inp=[loss_value], Tout=[])

                return loss_value, grads

            # store these information as members so we can use them outside the scope
            f.iter = tf.Variable(0)
            f.idx = idx
            f.part = part
            f.shapes = shapes
            f.assign_new_model_parameters = assign_new_model_parameters
            f.history = []

            return f

        inps = np.concatenate([x for x, y in dataset], axis=0)
        outs = np.concatenate([y for x, y in dataset], axis=0)
 
        # prepare prediction model, loss function, and the function passed to L-BFGS solver

        loss_fun = tf.keras.losses.MeanSquaredError()
        func = function_factory(self, loss_fun, inps, outs)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.network.trainable_variables)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params, max_iterations=2000, tolerance=1e-12)#, parallel_iterations=4)

        func.assign_new_model_parameters(results.position)
        self.history.history = func.history

    def model_size_stats(self):
        size_stats = {
            'params' : [count_nonzero_params(self.network)],
            'size' : [utils.get_gzipped_model_size(self)],
        }
        self.config.update(size_stats)

    def save(self, df_file):
        timestamp = pd.Timestamp(time.time(), unit='s').round('s').ctime()
        self.directory = os.path.abspath('.') +"/Data/Networks/"+ str(pd.Timestamp(timestamp).to_julian_date()) + "/"
        os.makedirs(self.directory, exist_ok=True)
        self.network.save(self.directory + "network")
        self.config['timetag'] = timestamp
        self.config['history'] = [self.history.history]
        self.config['id'] = [pd.Timestamp(timestamp).to_julian_date()]
        try:
            self.config['activation'] = [self.config['activation'][0].__name__]
        except:
            pass
        try:
            self.config['optimizer'] = [self.config['optimizer'][0].__module__]
        except:
            pass
        self.model_size_stats()
        utils.save_df_row(self.config, df_file)


def load_config_and_model(model_id, df_file):
    # Get the parameters and stats for a given run
    # If the dataframe hasn't been loaded
    if type(df_file) == str:
        config = utils.get_df_row(model_id, df_file)
    else:
        # If the dataframe has already been loaded
        config = df_file[model_id == df_file['id']].to_dict()
        for key, value in config.items():
            config[key] = list(value.values())

    # Reinitialize the model
    if 'mixed_precision' not in config:
        config['use_precision'] = [False]
    
    network = tf.keras.models.load_model(os.path.abspath('.') + "/Data/Networks/"+str(model_id)+"/network")
    model = CustomModel(config, network)
    if 'adam' in config['optimizer'][0]:
        optimizer = tf.keras.optimizers.Adam()
    elif 'rms' in config['optimizer'][0]:
        optimizer = tf.keras.optimizers.RMSprop()
    else:
        exit("No Optimizer Found")
    model.compile(optimizer=optimizer, loss='mse') #! Check that this compile is even necessary

    return config, model

def count_nonzero_params(model):
    params = 0
    for v in model.trainable_variables:
        params += tf.math.count_nonzero(v)
    return params.numpy()
