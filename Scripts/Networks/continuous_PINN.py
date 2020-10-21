
import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import pickle

from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, x1, x2, a0, a1, a2, layers, PINN=True, activation='tanh'):
        
        X = np.concatenate([x0, x1, x2], 1) # N x 3

        self.lb = X.min(0) # min of each components
        self.ub = X.max(0) # max of each components
        
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        
        self.layers = layers

        self.PINN = PINN
        self.activation = activation
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.x1_tf = tf.placeholder(tf.float32, shape=[None, self.x1.shape[1]])
        self.x2_tf = tf.placeholder(tf.float32, shape=[None, self.x2.shape[1]])
        
        self.a0_tf = tf.placeholder(tf.float32, shape=[None, self.a0.shape[1]])
        self.a1_tf = tf.placeholder(tf.float32, shape=[None, self.a1.shape[1]])
        self.a2_tf = tf.placeholder(tf.float32, shape=[None, self.a2.shape[1]])

        
        self.a0_pred, self.a1_pred, self.a2_pred, self.f_a0_pred, self.f_a1_pred, self.f_a2_pred, self.U_pred = self.net_NS(self.x0_tf, self.x1_tf, self.x2_tf)
        
        if self.PINN:
            self.loss = tf.reduce_sum(tf.square(self.a0_tf - self.a0_pred)) + \
                        tf.reduce_sum(tf.square(self.a1_tf - self.a1_pred)) + \
                        tf.reduce_sum(tf.square(self.a2_tf - self.a2_pred)) + \
                        tf.reduce_sum(tf.square(self.a0_tf + self.f_a0_pred)) + \
                        tf.reduce_sum(tf.square(self.a1_tf + self.f_a1_pred)) + \
                        tf.reduce_sum(tf.square(self.a2_tf + self.f_a2_pred))
        else:
            self.loss = tf.reduce_sum(tf.square(self.a0_tf - self.a0_pred)) + \
                        tf.reduce_sum(tf.square(self.a1_tf - self.a1_pred)) + \
                        tf.reduce_sum(tf.square(self.a2_tf - self.a2_pred))
 
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        #eps = 1E-6
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            if self.activation == 'tanh':
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            if self.activation == 'relu':
                H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
      
    def net_NS(self, x0, x1, x2):
       
        a_and_U = self.neural_net(tf.concat([x0,x1,x2], 1), self.weights, self.biases)
        a0 = a_and_U[:,0:1]
        a1 = a_and_U[:,1:2]
        a2 = a_and_U[:,2:3]

        U = a_and_U[:,3:4]
        
        if self.PINN: 
            U_x0 = tf.gradients(U, x0)[0]
            U_x1 = tf.gradients(U, x1)[0]
            U_x2 = tf.gradients(U, x2)[0]
        else:
            U_x0 = None
            U_x1 = None
            U_x2 = None

        return a0, a1, a2, U_x0, U_x1, U_x2, U
    
    def callback(self, loss):
        print('Loss:', loss)
    
    def train(self, nIter):
        tf_dict = {self.x0_tf: self.x0, 
                    self.x1_tf: self.x1, 
                    self.x2_tf: self.x2,
                    self.a0_tf: self.a0,
                    self.a1_tf: self.a1,
                    self.a2_tf: self.a2}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
    
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
    
    def predict(self, x0_star, x1_star, x2_star):
        tf_dict = {self.x0_tf: x0_star, 
                    self.x1_tf: x1_star, 
                    self.x2_tf: x2_star}

        a0_star = self.sess.run(self.a0_pred, tf_dict)
        a1_star = self.sess.run(self.a1_pred, tf_dict)
        a2_star = self.sess.run(self.a2_pred, tf_dict)
        U_star = self.sess.run(self.U_pred, tf_dict)

        return a0_star, a1_star, a2_star, U_star

    
if __name__ == "__main__": 

    configurations = {
        "config_4" : {
            'N_train' : 40000,
            'PINN' : True,
            'epochs' : 400000,
        },
        "config_5" : {
            'N_train' : 40000,
            'PINN' : False,
            'epochs' : 400000,
        },
        "config_2" : {
            'N_train' : 60000,
            'PINN' : False,
            'epochs' : 200000,
        },
        "config_3" : {
            'N_train' : 60000,
            'PINN' : True,
            'epochs' : 200000,
        },
        "config_6" : {
            'N_train' : 60000,
            'PINN' : True,
            'epochs' : 400000,
        },
        "config_1" : {
            'N_train' : 60000,
            'PINN' : False,
            'epochs' : 400000,
        },
    }    

    for key, value in configurations.items():
                
        N_train = value['N_train']

        layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 4]
        layers = [3, 40, 40, 40, 40, 40, 40, 40, 40, 4]

        #layers = [3, 100, 100, 100, 4]

        planet = Earth()
        model_file = planet.sh_hf_file
        density_deg = 180
        max_deg = 1000
        epochs = value['epochs']

        PINN = value['PINN']
        radius_min = planet.radius
        radius_max = planet.radius + 10

        activation = 'tanh'
        #activation = 'relu'

        df_file = "continuous_results.data"

        #trajectory = ReducedGridDist(planet, radius_min, degree=density_deg, reduction=0.25)
        #radius _max = None

        trajectory = ReducedRandDist(planet, [radius_min, radius_max], points=15488*4, degree=density_deg, reduction=0.25)
        map_trajectory = ReducedGridDist(planet, radius_min, degree=density_deg, reduction=0.25)
        
        # trajectory = DHGridDist(planet, radius_min, degree=density_deg)
        # map_trajectory = trajectory
        # radius_max = None

        Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
        accelerations = Call_r0_gm.load()

        C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
        accelerations_C22 = C22_r0_gm.load()

        #t = None #time T x 1
        x = Call_r0_gm.positions # position (N x 3)
        a = accelerations-accelerations_C22 # acceleration (N x 3) 
        u = None # potential (N x 1)

        x0 = x[:,0].reshape(-1,1) #r
        x1 = x[:,1].reshape(-1,1) #theta
        x2 = x[:,2].reshape(-1,1) #phi
        
        a0 = a[:,0].reshape(-1,1) #a r
        a1 = a[:,1].reshape(-1,1) #a theta
        a2 = a[:,2].reshape(-1,1) #a phi

        # Initial Data
        noise_a0 = 0.0
        idx_x = np.random.choice(x.shape[0], N_train, replace=False) 
        x0_train = x0[idx_x]
        x1_train = x1[idx_x]
        x2_train = x2[idx_x]

        a0_train = a0[idx_x]
        a1_train = a1[idx_x]
        a2_train = a2[idx_x]


        # Add Noise if interested
        # u0 = u0 + noise_a0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])

        model = PhysicsInformedNN(x0_train, x1_train, x2_train, 
                                    a0_train, a1_train, a2_train, 
                                    layers, PINN, activation)
        start = time.time()
        model.train(epochs)
        time_delta = np.round(time.time() - start, 2)
                    
        # Test data
        mask = np.ones(x0.shape,dtype=bool) #np.ones_like(a,dtype=bool)
        mask[idx_x] = False

        x0_star = x0[~mask].reshape(-1,1)
        x1_star = x1[~mask].reshape(-1,1)
        x2_star = x2[~mask].reshape(-1,1)

        a0_star = a0[~mask].reshape(-1,1)
        a1_star = a1[~mask].reshape(-1,1)
        a2_star = a2[~mask].reshape(-1,1)

        U_star = None
        
        a0_pred, a1_pred, a2_pred, U_pred = model.predict(x0_star, x1_star, x2_star)

        a_pred = np.hstack([a0_pred, a1_pred, a2_pred])
        a_test = np.hstack([a0_star, a1_star, a2_star])

        error = np.abs(np.divide(a_pred-a_test, a_test))*100
        print("Test Error")
        print(np.mean(error, axis=0))

        ######################################################################
        ############################# Stats ##################################
        ######################################################################    

        Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=map_trajectory)
        Call_a = Call_r0_gm.load()
        
        C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=map_trajectory)
        C22_a = C22_r0_gm.load()

        x = Call_r0_gm.positions # position (N x 3)
        a = Call_a-C22_a # acceleration (N x 3) 

        x0 = x[:,0].reshape(-1,1) #r
        x1 = x[:,1].reshape(-1,1) #theta
        x2 = x[:,2].reshape(-1,1) #phi

        a0_pred, a1_pred, a2_pred, U_pred = model.predict(x0, x1, x2)

        acc_pred = np.hstack((a0_pred, a1_pred, a2_pred))

        error = np.abs(np.divide((acc_pred - a), a))*100 # Percent Error for each component
        RSE_Call = np.sqrt(np.square(acc_pred - a))

        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        timestamp = pd.Timestamp(time.time(), unit='s').round('s').ctime()
        entries = {
            'timetag' : [timestamp],
            'trajectory' : [trajectory.__class__.__name__],
            'radius_min' : [radius_min],
            'radius_max' : [radius_max],
            'train_time' : [time_delta],
            'PINN_flag' : [PINN],
            'N_train' : [N_train],
            'degree' : [max_deg],
            'layers' : [layers],
            'epochs' : [epochs],
            'activation' : [activation],

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
        df = pd.DataFrame().from_dict(entries).set_index('timetag')
        print(df)

        try: 
            df_all = pd.read_pickle(df_file)
            df_all = df_all.append(df)
            df_all.to_pickle(df_file)
        except: 
            df.to_pickle(df_file)



        ######################################################################
        ############################# Plotting ###############################
        ######################################################################    


        grid_true = Grid(trajectory=map_trajectory, accelerations=a)
        grid_pred = Grid(trajectory=map_trajectory, accelerations=acc_pred)
        grid_C20 = Grid(trajectory=map_trajectory, accelerations=C22_a)

        # mapUnit = "m/s^2"
        mapUnit = 'mGal'
        map_vis = MapVisualization(mapUnit)
        directory = os.path.abspath('.') +"/Plots/"+ str(pd.Timestamp(timestamp).to_julian_date()) + "/"
        os.makedirs(directory, exist_ok=True)

        plt.rc('text', usetex=False)
        fig_true, ax = map_vis.plot_grid(grid_true.total, "True Grid [mGal]")

        map_vis.save(fig_true, directory + "true.pdf")
        fig_pred, ax = map_vis.plot_grid(grid_pred.total, "NN Grid [mGal]")
        map_vis.save(fig_pred, directory + "pred.pdf")

        diff = grid_pred - grid_true
        fig_pert, ax = map_vis.plot_grid(diff.total, "Acceleration Difference [mGal]")
        map_vis.save(fig_pert, directory + "diff.pdf")


        map_vis.fig_size = (5*4,3.5*4)
        fig, ax = map_vis.newFig()
        vlim = None 
        plt.subplot(311)
        im = map_vis.new_map(grid_true.total, vlim=None, log_scale=False)
        map_vis.add_colorbar(im, '[mGal]', vlim)
        
        plt.subplot(312)
        im = map_vis.new_map(grid_pred.total, vlim=None, log_scale=False)
        map_vis.add_colorbar(im, '[mGal]', vlim)
        
        plt.subplot(313)
        im = map_vis.new_map(diff.total, vlim=None, log_scale=False)
        map_vis.add_colorbar(im, '[mGal]', vlim)
        map_vis.save(fig, directory + "all.pdf")

        #plt.show()