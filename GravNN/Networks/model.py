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
#tf.set_random_seed(1234)

#tf.compat.v1.disable_eager_execution()
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class PhysicsInformedNN():
    # Initialize the class
    def __init__(self, config=None):
        self.config = config

        # Initialize NN              
        self.network = self.neural_net()

        # tfconfig = tf.ConfigProto()
        # tfconfig.gpu_options.allow_growth = True

        # self.sess = tf.Session(config=tfconfig)
        # self.sess.run(tf.global_variables_initializer())        
        #self.saver = tf.train.Saver(max_to_keep=None)

        self.loss = tf.keras.metrics.MeanSquaredError()
        self.adam_optimizer = tf.keras.optimizers.Adam()
        #self.network.compile(optimizer=self.adam_optimizer, loss=self.loss)

        
    def load_weights_biases(self, layers):        
        weights = []
        biases = []
        with open(os.path.abspath('.') +"/Plots/"+str(self.config['init_file'][0])+"/network.data", 'rb') as f:
            weights_init = pickle.load(f)
            biases_init = pickle.load(f)
        for l in range(0, len(layers)  - 1):
            weights.append(tf.Variable(weights_init[l], dtype=tf.float32))
            biases.append(tf.Variable(biases_init[l], dtype=tf.float32))  
        return weights, biases
    
    def neural_net(self):
        """
        Define the NN which acts as the solution to the PDE
        """
        if self.config['init_file'][0] is not None:
            model = self.load(self.config['init_file'][0])
        else:
            layers = self.config['layers'][0]
            # generate NN  
            self.inputs = tf.keras.Input(shape=(layers[0],))
            x = self.inputs
            for i in range(1,len(layers)-1):
                x = tf.keras.layers.Dense(units=layers[i], 
                                            activation=self.config['activation'][0], 
                                            kernel_initializer='glorot_normal')(x)
            self.outputs = tf.keras.layers.Dense(units=layers[-1], 
                                            activation='linear', 
                                            kernel_initializer='glorot_normal')(x)
            model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        return model

    def physics_constraints(self, x):
        # PINN Constrains
        if self.config['PINN_flag'][0]:
            assert self.network.layers[-1].output_shape[1] == 1
            with tf.GradientTape() as tape:
                tape.watch(x)
                U_pred = self.network(x)
            output = -tape.gradient(U_pred, x)
        else: 
            output = self.network(x)
            U_pred = tf.constant(0)#None

        return U_pred, output

    def predict(self, x_test):
        for x in x_test:
            return self.physics_constraints(x)
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            U_pred, y_pred = self.physics_constraints(x)
            dynamics_loss = tf.reduce_mean(tf.square((y - y_pred)))
            #representation_loss = tf.reduce_mean(tf.square((u - U_pred)))
            loss_result = dynamics_loss
        gradients = tape.gradient(loss_result, self.network.trainable_variables)
        self.adam_optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        #print(loss)
        return loss_result

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
                set_weights(self.network, params, sizes_w, sizes_b)
                U_pred, y_pred = self.physics_constraints(x)
                dynamics_loss = tf.reduce_mean(tf.square((y - y_pred)))
            print(dynamics_loss.numpy())
            gradients = tape.gradient(dynamics_loss, self.network.trainable_variables)
            grad_flat = flatten_variables(gradients)
            return dynamics_loss, grad_flat
    
        # # SciPy optimization
        params = flatten_variables(self.network.trainable_variables)
        results = tfp.optimizer.lbfgs_minimize(loss_and_gradient,
                                                 params, 
                                                 max_iterations=5000,
                                                 parallel_iterations=multiprocessing.cpu_count(),
                                                 x_tolerance=1.0*np.finfo(float).eps)
        print(results.converged)
        set_weights(self.network, results.position, sizes_w, sizes_b)
    
    #@tf.function
    def train(self, dataset, epochs, batch_size):
        # SGD optimization
        start = time.time()
        for epoch in range(epochs):
            loss = 0.0
            for step, (x_batch_train, y_batch_train) in enumerate(dataset):
                loss += self.train_step(x_batch_train, y_batch_train)
            if epoch % 10 == 0:
                print(
                    f'Epoch {epoch}, '
                    f'Loss: {loss.numpy():.{4}}, '
                    f'Time: {time.time()-start:.{4}}, ')
                start = time.time()
        
       


    def save(self, path):
        self.network.save(path)
        
    def load(self, path):
        return tf.keras.models.load_model(path)
