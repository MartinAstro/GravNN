
import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from GravNN.GravityModels.NN_Base import NN_Base

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, x1, x2, a0, a1, a2, layers):
        
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
        
        self.loss = tf.reduce_sum(tf.square(self.a0_tf - self.a0_pred)) + \
                    tf.reduce_sum(tf.square(self.a1_tf - self.a1_pred)) + \
                    tf.reduce_sum(tf.square(self.a2_tf - self.a2_pred)) + \
                    tf.reduce_sum(tf.square(self.a0_tf + self.f_a0_pred)) + \
                    tf.reduce_sum(tf.square(self.a1_tf + self.f_a1_pred)) + \
                    tf.reduce_sum(tf.square(self.a2_tf + self.f_a2_pred))
                    # tf.reduce_sum(tf.square(self.f_a0_pred)) + \
                    # tf.reduce_sum(tf.square(self.f_a1_pred)) + \
                    # tf.reduce_sum(tf.square(self.f_a2_pred))

 
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
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            #H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
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
        
        U_x0 = tf.gradients(U, x0)[0]
        U_x1 = tf.gradients(U, x1)[0]
        U_x2 = tf.gradients(U, x2)[0]

        # f_a0 = a0 + U_x0 # dx^2/dt^2 = -U_x
        # f_a1 = a1 + U_x1
        # f_a2 = a2 + U_x2
        
        f_a0 = U_x0 # dx^2/dt^2 = -U_x
        f_a1 = U_x1
        f_a2 = U_x2

        return a0, a1, a2, f_a0, f_a1, f_a2, U
    
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
        
    N_train = 40000
    
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 4]
    #layers = [3, 100, 100, 100, 4]

    planet = Earth()
    radius = planet.radius
    model_file = planet.sh_hf_file
    density_deg = 175
    max_deg = 4

    #trajectory = ReducedGridDist(planet, radius, degree=density_deg, reduction=0.25)
    trajectory = ReducedRandDist(planet, [radius, radius+10], points=15488*4, degree=density_deg, reduction=0.25)
    #trajectory = DHGridDist(planet, radius, degree=density_deg)

    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    accelerations = Call_r0_gm.load()

    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
    accelerations_C22 = C22_r0_gm.load()
    #Call_r0_grid = Grid(trajectory=trajectory, accelerations=Call_r0_gm.load())

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
                                layers)
    model.train(200000)

                
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

    error_a0 = np.linalg.norm(a0_star-a0_pred,2)/np.linalg.norm(a0_star,2)*100
    error_a1 = np.linalg.norm(a1_star-a1_pred,2)/np.linalg.norm(a1_star,2)*100
    error_a2 = np.linalg.norm(a2_star-a2_pred,2)/np.linalg.norm(a2_star,2)*100

    print('Error a0: %e Percent' % (error_a0))    
    print('Error a1: %e Percent' % (error_a1))    
    print('Error a2: %e Percent' % (error_a2))   

   

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    trajectory = ReducedGridDist(planet, radius, degree=density_deg, reduction=0.25)
    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    accelerations = Call_r0_gm.load()
    
    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
    accelerations_C22 = C22_r0_gm.load()

    x = Call_r0_gm.positions # position (N x 3)
    a = accelerations-accelerations_C22 # acceleration (N x 3) 

    x0 = x[:,0].reshape(-1,1) #r
    x1 = x[:,1].reshape(-1,1) #theta
    x2 = x[:,2].reshape(-1,1) #phi

    a0_pred, a1_pred, a2_pred, U_pred = model.predict(x0, x1, x2)

    acc_pred = np.hstack((a0_pred, a1_pred, a2_pred))

    grid_true = Grid(trajectory=trajectory, accelerations=a)
    grid_pred = Grid(trajectory=trajectory, accelerations=acc_pred)
    grid_C20 = Grid(trajectory=trajectory, accelerations=accelerations_C22)


    # mapUnit = "m/s^2"
    mapUnit = 'mGal'
    map_vis = MapVisualization(mapUnit)#MapVisualization(save_directory=os.path.splitext(__file__)[0]  + "/../../../Plots/" +os.path.splitext(os.path.basename(__file__))[0] + "/",halt_formatting=True)
    plt.rc('text', usetex=False)
    fig_true = map_vis.plot_grid(grid_true.total, "True Grid [mGal]")
    fig_pred = map_vis.plot_grid(grid_pred.total, "NN Grid [mGal]")

    # fig_true_c22 = map_vis.plot_grid((grid_true-grid_C20).total, "True Grid - C22 [mGal]")
    # fig_pred_c22 = map_vis.plot_grid((grid_pred-grid_C20).total, "NN Grid - C22[mGal]")

    diff = grid_pred - grid_true
    fig_pert, ax = map_vis.plot_grid(diff.total, "Acceleration Difference [mGal]")

    plt.show()

