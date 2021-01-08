import os
import pickle
import sys
import time
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
import tensorflow_probability as tfp

np.random.seed(1234)

class CustomModel(tf.keras.Model):
    # Initialize the class
    def __init__(self, config, network):
        super(CustomModel, self).__init__()
        self.config = config
        self.network = network

    @tf.function
    def call(self, x, training=None):
        # PINN Constraints
        if self.config['PINN_flag'][0]:
            # if self.config['basis'][0] == 'spherical':
            #     # This cannot work as currently designed. The gradient at theta [0, 180] is divergent. 
            #     with tf.GradientTape() as tape:
            #         tape.watch(x)
            #         U_pred = self.network(x, training)
            #     gradients = tape.gradient(U_pred, x)
            #     # https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates#Del_formula
            #     a0 = -gradients[:,0]
            #     # In wiki article, theta is 0-180 deg (which has been phi in our definition)
            #     theta = tf.add(tf.multiply(x[:,2],np.pi), np.pi)
            #     a1 = -(1.0/x[:,0])*(1.0/tf.sin(theta))*gradients[:,1]
            #     a2 = -(1.0/x[:,0])*gradients[:,2]

            #     #print(a2.shape)
            #     a_pred = tf.concat([[a0], [a1], [a2]], 0)
            #     a_pred = tf.reshape(a_pred, [-1, 3])
            # else:                
            with tf.GradientTape() as tape:
                tape.watch(x)
                U_pred = self.network(x, training)
            a_pred = -tape.gradient(U_pred, x)
        else:  
            a_pred = self.network(x, training)
            U_pred = tf.constant(0.0)#None
        return U_pred, a_pred
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            U_pred, a_pred = self(x, training=True)
            #a_pred = tf.where(tf.math.is_inf(a_pred), y, a_pred)
            loss = self.compiled_loss(y, a_pred)#, regularization_losses=self.losses)
            
            # TODO: Put in mixed precision training
            scaled_loss = self.optimizer.get_scaled_loss(loss)


            # # Periodic boundary conditions 
            # if self.config['basis'][0] == 'spherical':
                
            #     x_periodic = tf.add(x, [0, 2, 2])
            #     U_pred_periodic, a_pred_periodic = self(x_periodic, training=True)
            #     #a_pred_periodic = tf.where(tf.math.is_inf(a_pred_periodic), y, a_pred_periodic)
            #     loss += self.compiled_loss(y, a_pred_periodic)

            #     x_periodic = tf.add(x, [0, -2, -2])
            #     U_pred_periodic, a_pred_periodic = self(x_periodic, training=True)
            #     #a_pred_periodic = tf.where(tf.math.is_inf(a_pred_periodic), y, a_pred_periodic)
            #     loss += self.compiled_loss(y, a_pred_periodic)

            #     # 0 potential at infinity. 
            #     x_infinite = tf.multiply(x, [1E308, 1, 1])
            #     U_pred_infinite, a_pred_infinite = self(x_infinite, training=True)
            #     a_pred_infinite = tf.where(tf.math.is_inf(a_pred_infinite), y, a_pred_infinite)
            #     a_pred_infinite = tf.where(tf.math.is_nan(a_pred_infinite), y, a_pred_infinite)
            #     loss += self.compiled_loss(tf.zeros_like(a_pred_infinite), a_pred_infinite)

        # TODO: Put in mixed precision training
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)

        #gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, a_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        
    @tf.function
    def test_step(self, data):
        x, y = data
        U_pred, a_pred = self(x, training=False)
        loss = self.compiled_loss(y, a_pred)#, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, a_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def optimize(self, dataset):
        
        class History:
            def __init__(self):
                self.history = []
        
        self.history = History()
        #L-BFGS Optimization
        x = np.concatenate([x for x, y in dataset], axis=0)
        y = np.concatenate([y for x, y in dataset], axis=0)

        sizes_w = []
        sizes_b = []
        for i, layer in enumerate(self.layers[0].layers):
            if i != 0 and not 'dropout' in layer.name:
                weights = layer.kernel.shape[0]
                for j in range(1,len(layer.kernel.shape)):
                    weights *= layer.kernel.shape[j]
                sizes_w.append(int(weights))
                sizes_b.append(int(layer.bias.shape[0]))

        def set_weights(model, w, sizes_w, sizes_b):
            i = 0
            for layer in model.layers[0].layers[1:]:
                if 'dropout' in layer.name:
                    continue
                start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
                end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
                weights = w[start_weights:end_weights]
                w_div = int(sizes_w[i] / sizes_b[i])
                weights = tf.reshape(weights, [w_div, sizes_b[i]])
                biases = w[end_weights:end_weights + sizes_b[i]]
                weights_biases = [weights, biases]
                layer.set_weights(weights_biases)
                i += 1

        def get_weights(model):
            w = []
            for layer in model.layers[0:]:
                weights_biases = layer.get_weights()
                weights = weights_biases[0].flatten()
                biases = weights_biases[1]
                w.extend(weights)
                w.extend(biases)
            w = tf.convert_to_tensor(w)
            return w
                
        def flatten_variables(variables):
            variables_flat = []
            for v in variables:
                # The gradient with respect to the final bias is non existant TODO: Figure out why. 
                if v is None:
                    v = 0.0
                variables_flat.append(tf.reshape(v, [-1]))
            variables_flat = tf.concat(variables_flat, 0)
            return variables_flat

        @tf.function
        def loss_and_gradient(params):
            with tf.GradientTape() as tape:
                U_pred, y_pred = self(x)
                y_pred = tf.where(tf.math.is_inf(y_pred), y, y_pred)
                dynamics_loss = tf.reduce_mean(tf.square((y - y_pred)))
            gradients = tape.gradient(dynamics_loss, self.trainable_variables)
            return dynamics_loss, gradients 

        def lgbfgs_loss_and_gradient(params):
            set_weights(self, params, sizes_w, sizes_b)
            loss, gradients = loss_and_gradient(params)
            print(loss.numpy())
            self.history.history.append(loss.numpy())
            grad_flat = flatten_variables(gradients)
            return loss, grad_flat
    
        # # SciPy optimization
        params = flatten_variables(self.trainable_variables)
        start_time = time.time()
        results = tfp.optimizer.lbfgs_minimize(lgbfgs_loss_and_gradient,
                                                 params, 
                                                 max_iterations=self.config['optimize_epochs'][0])#,
                                                 #parallel_iterations=multiprocessing.cpu_count(),
                                                 #x_tolerance=1.0*np.finfo(float).eps)
        print("Converged: " + str(results.converged) +" ; In " + str(time.time() - start_time) + " seconds")
        set_weights(self, results.position, sizes_w, sizes_b)

    
    def model_size_stats(self):
        size_stats = {
            'params' : [utils.count_nonzero_params(self.network)],
            'size' : [utils.get_gzipped_model_size(self)],
        }
        self.config.update(size_stats)

    def save(self, df_file):
        timestamp = pd.Timestamp(time.time(), unit='s').round('s').ctime()
        self.directory = os.path.abspath('.') +"/Plots/"+ str(pd.Timestamp(timestamp).to_julian_date()) + "/"
        os.makedirs(self.directory, exist_ok=True)
        self.network.save(self.directory + "network")
        self.config['timetag'] = timestamp
        self.config['history'] = [self.history.history]
        self.config['id'] = [pd.Timestamp(timestamp).to_julian_date()]
        self.model_size_stats()
        utils.save_df_row(self.config, df_file)


def load_config_and_model(model_id, df_file):
    # Get the parameters and stats for a given run
    config = utils.get_df_row(model_id, df_file)

    # Reinitialize the model
    network = tf.keras.models.load_model(os.path.abspath('.') +"/Plots/"+str(model_id)+"/network")
    model = CustomModel(config, network)
    model.compile(optimizer=config['optimizer'][0], loss='mse') #! Check that this compile is even necessary

    return config, model