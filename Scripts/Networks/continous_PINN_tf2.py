
import os
import pickle
import sys
import time

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

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, x1, x2, a0, a1, a2, config=None):
        
        X = np.concatenate([x0, x1, x2], 1) # N x 3

        self.lb = X.min(0) # min of each components
        self.ub = X.max(0) # max of each components
        
        self.config = config

        # Initialize NN
        if self.config['init_file'][0] is not None:
            self.weights, self.biases = self.load_weights_biases(self.config['layers'][0])  
        else:
            self.weights, self.biases = None, None
        
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, x0.shape[1]])
        self.x1_tf = tf.placeholder(tf.float32, shape=[None, x1.shape[1]])
        self.x2_tf = tf.placeholder(tf.float32, shape=[None, x2.shape[1]])
        
        self.a0_tf = tf.placeholder(tf.float32, shape=[None, a0.shape[1]])
        self.a1_tf = tf.placeholder(tf.float32, shape=[None, a1.shape[1]])
        self.a2_tf = tf.placeholder(tf.float32, shape=[None, a2.shape[1]])
        
        self.a0_pred, self.a1_pred, self.a2_pred, self.U_pred = self.f_model(self.x0_tf, self.x1_tf, self.x2_tf)

        def loss_fn(y_true, y_pred):
            return tf.reduce_sum(tf.square(y_true - y_pred))

        self.loss = loss_fn
        '''
        tf.reduce_sum(tf.square(self.a0_tf - self.a0_pred)) + \
                    tf.reduce_sum(tf.square(self.a1_tf - self.a1_pred)) + \
                    tf.reduce_sum(tf.square(self.a2_tf - self.a2_pred))
        '''
        self.network.compile(optimizer='adam',
                                loss=self.loss,
                                metrics=['accuracy'])


    def f_model(self, x0, x1, x2):

        self.network = self.neural_net(tf.concat([x0,x1,x2], 1), self.weights, self.biases)
        if self.config['PINN_flag'][0]:
            U_pred = self.network(tf.concat([x0, x1, x2], 1))
            a0_pred = -tf.gradients(U_pred, x0)
            a1_pred = -tf.gradients(U_pred, x1)
            a2_pred = -tf.gradients(U_pred, x2)
            # with tf.GradientTape(persistent=True) as tape:
            #     tape.watch(x)
            #     tape.watch(t)

            #     U_pred = self.self.network(tf.concat([x0, x1, x2], 1))
            #     a0_pred = -tape.gradients(U_pred, x0)
            #     a1_pred = -tape.gradients(U_pred, x1)
            #     a2_pred = -tape.gradients(U_pred, x2)
            # del tape
        else:
            acc = self.network(tf.concat([x0,x1,x2], 1))

            U_pred = None
            a0_pred = acc[:,0:1]
            a1_pred = acc[:,1:2]
            a2_pred = acc[:,2:3]       

        return a0_pred, a1_pred, a2_pred, U_pred
        
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
           
    def neural_net(self, X, weights=None, biases=None):
        layers = self.config['layers'][0]

        inputs = [tf.keras.layers.InputLayer(input_shape=(layers[0],))]
        hidden_layers = []
        for i in range(len(layers)-1):
            hidden_layer = tf.keras.layers.Dense(
                                            units=layers[i], 
                                            activation=self.config['activation'][0], 
                                            kernel_initializer='glorot_normal')
            hidden_layers.append(hidden_layer)
                                                    
        outputs = [tf.keras.layers.Dense(
                                    units=layers[-1], 
                                    activation='linear', 
                                    kernel_initializer='glorot_normal')]

        model = tf.keras.Sequential(
            np.concatenate(
                    (inputs,
                    hidden_layers,
                    outputs)).tolist()
        )
        if weights is not None:
            model.set_weights(weights)
        if biases is not None: 
            model.set_biases(biases)
        return model

    def optimize(self):
        self.optimizer = fmin_l_bfgs_b(self.loss, tf.get_weights(self.network), tf.gradient(self.loss, tf.concat([self.x0_tf, self.x1_tf, self.x2_tf])), 
                                        maxfun=50000, 
                                        maxiter=50000,
                                        epsilon=1.0*np.finfo(float).eps, 
                                        maxls=50) 
                                        
        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
        #                                                         method = 'L-BFGS-B', 
        #                                                         options = {'maxiter': 50000,
        #                                                                    'maxfun': 50000,
        #                                                                    'maxcor': 50,
        #                                                                    'maxls': 50,
        #                                                                    'ftol' : 1.0 * np.finfo(float).eps})
    def predict(self, x0, x1, x2):
        return self.f_model(tf.concat([x0, x1, x2]))
    
def main():
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000
    save = True
    train = True
    
    configurations = {
        "config_nonPINN" : {
            'N_train' : [40000],
            'PINN_flag' : [False],
            'epochs' : [200000], 
            'radius_max' : [planet.radius + 10.0],
            'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
            'acc_noise' : [0.00],
            'deg_removed' : [2],
            'activation' : ['tanh'],
            'init_file': [None],
            'notes' : ['nonPINN - No potential included']
        },
    }    

    for key, config in configurations.items():
        
        tf.reset_default_graph()

        radius_min = planet.radius

        df_file = "continuous_results.data"

        # trajectory = ReducedRandDist(planet, [radius_min, config['radius_max'][0]], points=15488*4, degree=density_deg, reduction=0.25)
        # map_trajectory = ReducedGridDist(planet, radius_min, degree=density_deg, reduction=0.25)

        trajectory = RandomDist(planet, [radius_min, config['radius_max'][0]], points=259200)
        map_trajectory =  DHGridDist(planet, radius_min, degree=density_deg)

        Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
        accelerations = Call_r0_gm.load()

        Clm_r0_gm = SphericalHarmonics(model_file, degree=int(config['deg_removed'][0]), trajectory=trajectory)
        accelerations_Clm = Clm_r0_gm.load()

        x_unscaled = Call_r0_gm.positions # position (N x 3)
        a_unscaled = accelerations - accelerations_Clm
        u = None # potential (N x 1)

        # Preprocessing
        x_transformer = MinMaxScaler(feature_range=(-1,1))
        x = x_transformer.fit_transform(x_unscaled)

        a_transformer = MinMaxScaler(feature_range=(-1,1))
        a = a_transformer.fit_transform(a_unscaled)
        
        # Initial Data
        idx_x = np.random.choice(x.shape[0], config['N_train'][0], replace=False) 

        x0_train = x[:,0].reshape(-1,1)[idx_x] #r
        x1_train = x[:,1].reshape(-1,1)[idx_x] #theta
        x2_train = x[:,2].reshape(-1,1)[idx_x] #phi

        a0_train = a[:,0].reshape(-1,1)[idx_x] #a r
        a1_train = a[:,1].reshape(-1,1)[idx_x] #a theta
        a2_train = a[:,2].reshape(-1,1)[idx_x] #a phi

        # Add Noise if interested
        a0_train = a0_train + config['acc_noise'][0]*np.std(a0_train)*np.random.randn(a0_train.shape[0], a0_train.shape[1])
        a1_train = a1_train + config['acc_noise'][0]*np.std(a1_train)*np.random.randn(a1_train.shape[0], a1_train.shape[1])
        a2_train = a2_train + config['acc_noise'][0]*np.std(a2_train)*np.random.randn(a2_train.shape[0], a2_train.shape[1])

        PINN = PhysicsInformedNN(x0_train, x1_train, x2_train, 
                                    a0_train, a1_train, a2_train, 
                                    config)

        start = time.time()
        if train:
            PINN.network.fit(x_train,
                        a_train,
                        epochs=config['epochs'][0],
                        validation_split=0.0)
            PINN.optimize()
        time_delta = np.round(time.time() - start, 2)
                    

        ######################################################################
        ############################# Training Stats #########################
        ######################################################################    

        Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=map_trajectory)
        Call_a = Call_r0_gm.load()
        
        Clm_r0_gm = SphericalHarmonics(model_file, degree=int(config['deg_removed'][0]), trajectory=map_trajectory)
        Clm_a = Clm_r0_gm.load()

        x = Call_r0_gm.positions # position (N x 3)
        a = Call_a - Clm_a

        x = x_transformer.transform(x)
        a = a_transformer.transform(a)
    
        x0 = x[:,0].reshape(-1,1) #r
        x1 = x[:,1].reshape(-1,1) #theta
        x2 = x[:,2].reshape(-1,1) #phi

        a0_pred, a1_pred, a2_pred, U_pred = PINN.predict(x0, x1, x2)
        acc_pred = np.hstack((a0_pred, a1_pred, a2_pred))

        x = x_transformer.inverse_transform(x)
        a = a_transformer.inverse_transform(a)
        a_pred = a_transformer.inverse_transform(a_pred)

        error = np.abs(np.divide((acc_pred - a), a))*100 # Percent Error for each component
        RSE_Call = np.sqrt(np.square(acc_pred - a))

        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        timestamp = pd.Timestamp(time.time(), unit='s').round('s').ctime()
        entries = {
            'timetag' : [timestamp],
            'trajectory' : [trajectory.__class__.__name__],
            'radius_min' : [radius_min],
            'train_time' : [time_delta],
            'degree' : [max_deg],

            'rse_mean' : [np.mean(RSE_Call)],
            'rse_std' : [np.std(RSE_Call)],
            'rse_median' : [np.median(RSE_Call)],
            'rse_a0_mean' : [np.mean(RSE_Call[:,0])],
            'rse_a1_mean' : [np.mean(RSE_Call[:,1])],
            'rse_a2_mean' : [np.mean(RSE_Call[:,2])],

            'percent_rel_mean' : [np.mean(error)],
            'percent_rel_std' : [np.std(error)], 
            'percent_rel_median' : [np.median(error)],
            'percent_rel_a0_mean' : [np.mean(error[:,0])], 
            'percent_rel_a1_mean' : [np.mean(error[:,1])], 
            'percent_rel_a2_mean' : [np.mean(error[:,2])],

            'params' : [params]
        }
        config.update(entries)

        ######################################################################
        ############################# Testing Stats ##########################
        ######################################################################    

        grid_true = Grid(trajectory=map_trajectory, accelerations=a)
        grid_pred = Grid(trajectory=map_trajectory, accelerations=acc_pred)
        diff = grid_pred - grid_true
       
        # This ensures the same features are being evaluated independent of what degree is taken off at beginning
        C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=map_trajectory)
        C22_a = C22_r0_gm.load()
        grid_C22 = Grid(trajectory=map_trajectory, accelerations=Call_a - C22_a)

        two_sigma_mask = np.where(grid_C22.total > (np.mean(grid_C22.total) + 2*np.std(grid_C22.total)))
        two_sigma_mask_compliment = np.where(grid_C22.total < (np.mean(grid_C22.total) + 2*np.std(grid_C22.total)))
        two_sig_features = diff.total[two_sigma_mask]
        two_sig_features_comp = diff.total[two_sigma_mask_compliment]

        three_sigma_mask = np.where(grid_C22.total > (np.mean(grid_C22.total) + 3*np.std(grid_C22.total)))
        three_sigma_mask_compliment = np.where(grid_C22.total < (np.mean(grid_C22.total) + 3*np.std(grid_C22.total)))
        three_sig_features = diff.total[three_sigma_mask]
        three_sig_features_comp = diff.total[three_sigma_mask_compliment]

        map_stats = {
            'sigma_2_mean' : [np.mean(np.sqrt(np.square(two_sig_features)))],
            'sigma_2_std' : [np.std(np.sqrt(np.square(two_sig_features)))],
            'sigma_2_median' : [np.median(np.sqrt(np.square(two_sig_features)))],

            'sigma_2_c_mean' : [np.mean(np.sqrt(np.square(two_sig_features_comp)))],
            'sigma_2_c_std' : [np.std(np.sqrt(np.square(two_sig_features_comp)))],
            'sigma_2_c_median' : [np.median(np.sqrt(np.square(two_sig_features_comp)))],

            'sigma_3_mean' : [np.mean(np.sqrt(np.square(three_sig_features)))],
            'sigma_3_std' : [np.std(np.sqrt(np.square(three_sig_features)))],
            'sigma_3_median' : [np.median(np.sqrt(np.square(three_sig_features)))],

            'sigma_3_c_mean' : [np.mean(np.sqrt(np.square(three_sig_features_comp)))],
            'sigma_3_c_std' : [np.std(np.sqrt(np.square(three_sig_features_comp)))],
            'sigma_3_c_median' : [np.median(np.sqrt(np.square(three_sig_features_comp)))],

            'max_error' : [np.max(np.sqrt(np.square(diff.total)))]
        }
        config.update(map_stats)
        df = pd.DataFrame().from_dict(config).set_index('timetag')

        ######################################################################
        ############################# Plotting ###############################
        ######################################################################    

        mapUnit = 'mGal'
        map_vis = MapVisualization(mapUnit)
        plt.rc('text', usetex=False)

        fig_true, ax = map_vis.plot_grid(grid_true.total, "True Grid [mGal]")
        fig_pred, ax = map_vis.plot_grid(grid_pred.total, "NN Grid [mGal]")
        fig_pert, ax = map_vis.plot_grid(diff.total, "Acceleration Difference [mGal]")

        map_vis.fig_size = (5*4,3.5*4)
        fig, ax = map_vis.newFig()
        vlim = [0, np.max(grid_true.total)*10000.0] 
        plt.subplot(311)
        im = map_vis.new_map(grid_true.total, vlim=vlim, log_scale=False)
        map_vis.add_colorbar(im, '[mGal]', vlim)
        
        plt.subplot(312)
        im = map_vis.new_map(grid_pred.total, vlim=vlim, log_scale=False)
        map_vis.add_colorbar(im, '[mGal]', vlim)
        
        plt.subplot(313)
        im = map_vis.new_map(diff.total, vlim=vlim, log_scale=False)
        map_vis.add_colorbar(im, '[mGal]', vlim)

        ######################################################################
        ############################# Saving #################################
        ######################################################################    

        if save: 
            try: 
                df_all = pd.read_pickle(df_file)
                df_all = df_all.append(df)
                df_all.to_pickle(df_file)
            except: 
                df.to_pickle(df_file)

            directory = os.path.abspath('.') +"/Plots/"+ str(pd.Timestamp(timestamp).to_julian_date()) + "/"
            os.makedirs(directory, exist_ok=True)

            map_vis.save(fig_true, directory + "true.pdf")
            map_vis.save(fig_pred, directory + "pred.pdf")
            map_vis.save(fig_pert, directory + "diff.pdf")
            map_vis.save(fig, directory + "all.pdf")

            with open(directory + "network.data", 'wb') as f:
                weights = []
                biases = []
                for i in range(len(model.weights)):
                    weights.append(model.weights[i].eval(session=model.sess))
                    biases.append(model.biases[i].eval(session=model.sess))
                pickle.dump(weights, f)
                pickle.dump(biases, f)
        
        model.sess.close()
        plt.close()
        #plt.show()

if __name__ == '__main__':
    main()