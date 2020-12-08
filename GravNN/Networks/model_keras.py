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
from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize, fmin_l_bfgs_b
import tensorflow_probability as tfp

np.random.seed(1234)

class CustomModel(tf.keras.Model):
    # # Initialize the class
    # def __init__(self, config, network):
    #     super(CustomModel, self).__init__()
    #     self.config = config
    #     self.network = network

    def set_config(self, config):
        self.config = config

    def physics_constraints(self, x, training=None):
        # PINN Constraints
        if self.config['PINN_flag'][0]:
            assert self.layers[-1].output_shape[1] == 1
            with tf.GradientTape() as tape:
                tape.watch(x)
                U_pred = self(x, training)
            a_pred = -tape.gradient(U_pred, x)
        else:  
            assert self.layers[-1].output_shape[1] == 3
            a_pred = self(x)
            U_pred = tf.constant(0.0)#None
        return U_pred, a_pred
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            U_pred, a_pred = self.physics_constraints(x, training=True)
            loss = self.compiled_loss(y, a_pred)#, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, a_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        
    
    @tf.function
    def predict_step(self, x):
        U_pred, a_pred = self.physics_constraints(x, training=False)
        return U_pred, a_pred

    def optimize(self, dataset):
        #L-BFGS Optimization
        x = np.concatenate([x for x, y in dataset], axis=0)
        y = np.concatenate([y for x, y in dataset], axis=0)

        sizes_w = []
        sizes_b = []
        for i, width in enumerate(self.config['layers'][0]):
            if i != 1:
                sizes_w.append(int(width * self.config['layers'][0][1]))
                sizes_b.append(int(width if i != 0 else self.config['layers'][0][1]))


        def set_weights(model, w, sizes_w, sizes_b):
            for i, layer in enumerate(model.layers[1:]):
                start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
                end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
                weights = w[start_weights:end_weights]
                w_div = int(sizes_w[i] / sizes_b[i])
                weights = tf.reshape(weights, [w_div, sizes_b[i]])
                biases = w[end_weights:end_weights + sizes_b[i]]
                weights_biases = [weights, biases]
                layer.set_weights(weights_biases)

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
                variables_flat.append(tf.reshape(v, [-1]))
            variables_flat = tf.concat(variables_flat, 0)
            return variables_flat

        
        def loss_and_gradient(params):
            with tf.GradientTape() as tape:
                set_weights(self, params, sizes_w, sizes_b)
                U_pred, y_pred = self.physics_constraints(x)
                dynamics_loss = tf.reduce_mean(tf.square((y - y_pred)))
            print(dynamics_loss.numpy())
            gradients = tape.gradient(dynamics_loss, self.trainable_variables)
            grad_flat = flatten_variables(gradients)
            return dynamics_loss, grad_flat
    
        # # SciPy optimization
        params = flatten_variables(self.trainable_variables)
        results = tfp.optimizer.lbfgs_minimize(loss_and_gradient,
                                                 params, 
                                                 max_iterations=50000,
                                                 parallel_iterations=multiprocessing.cpu_count(),
                                                 x_tolerance=1.0*np.finfo(float).eps)
        print("Converged: " + str(results.converged))
        set_weights(self, results.position, sizes_w, sizes_b)

        
