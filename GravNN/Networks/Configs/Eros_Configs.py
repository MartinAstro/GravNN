from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Trajectories import RandomAsteroidDist
from GravNN.Preprocessors import DummyScaler, UniformScaler
from sklearn.preprocessing import MinMaxScaler
def get_prototype_eros_config():
    data_config = {
        'planet' : [Eros()],
        'grav_file' : [Eros().model_potatok],
        'distribution' : [RandomAsteroidDist],
        'N_dist' : [10000],
        'N_train' : [2500], 
        'N_val' : [150],
        'radius_min' : [0],
        'radius_max' : [Eros().radius + 5000.0],
        'acc_noise' : [0.00],
        'basis' : [None],
        'deg_removed' : [0],
        'mixed_precision' : [False],
        'dtype' :['float32'],
        'max_deg' : [1000], 
        'analytic_truth' : ['poly_stats_']
    }
    network_config = {
        'network_type' : ['traditional'],
        'PINN_constraint_fcn' : ['no_pinn'],
        'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
        'activation' : ['gelu'],
        'init_file' : [None],
        'epochs' : [100000],
        'initializer' : ['glorot_normal'],
        'optimizer' : ['adam'],
        'batch_size' : [131072 // 2],
        'dropout' : [0.0], 
        'x_transformer' : [UniformScaler(feature_range=(-1,1))],
        'u_transformer' : [UniformScaler(feature_range=(-1,1))],
        'a_transformer' : [UniformScaler(feature_range=(-1,1))],
        'dtype' : ['float32'],
        'dummy_transformer' : [DummyScaler()],
        'class_weight' : [[1.0, 1.0, 1.0]] #no_pinn and PINN_A
    }

    config = {}
    config.update(data_config)
    config.update(network_config)
    return config

def get_default_eros_config():
    data_config = {
        'planet' : [Eros()],
        'grav_file' : [Eros().model_25k],
        'distribution' : [RandomAsteroidDist],
        'N_dist' : [100000],
        'N_train' : [95000], 
        'N_val' : [5000],
        'radius_min' : [0],
        'radius_max' : [Eros().radius + 10000.0],
        'acc_noise' : [0.00],
        'basis' : [None],
        'deg_removed' : [0],
        'mixed_precision' : [False],
        'dtype' :['float32'],
        'max_deg' : [1000], 
        'analytic_truth' : ['poly_stats_']
    }
    network_config = {
        'network_type' : ['traditional'],
        'PINN_constraint_fcn' : ['no_pinn'],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
        'activation' : ['tanh'],
        'init_file' : [None],
        'epochs' : [100000],
        'initializer' : ['glorot_normal'],
        'optimizer' : ['adam'],
        'batch_size' : [40000],
        'dropout' : [0.0], 
        'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'a_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'dtype' : ['float32'],
        'dummy_transformer' : [DummyScaler()],
        'class_weight' : [[1.0, 1.0, 1.0]] #no_pinn and PINN_A
    }

    config = {}
    config.update(data_config)
    config.update(network_config)
    return config

def get_default_eros_pinn_config():
    data_config = {
        'planet' : [Eros()],
        'grav_file' : [Eros().model_25k],
        'distribution' : [RandomAsteroidDist],
        'N_dist' : [100000],
        'N_train' : [95000], 
        'N_val' : [5000],
        'radius_min' : [0],
        'radius_max' : [Eros().radius + 10000.0],
        'acc_noise' : [0.00],
        'basis' : [None],
        'deg_removed' : [0],
        'mixed_precision' : [False],
        'dtype' :['float32'],
        'max_deg' : [1000], 
        'analytic_truth' : ['poly_stats_']
    }
    network_config = {
        'network_type' : ['traditional'],
        'PINN_constraint_fcn' : ['pinn_A'],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 1]],
        'activation' : ['tanh'],
        'init_file' : [None],
        'epochs' : [100000],
        'optimizer' : ['adam'],
        'batch_size' : [40000],
        'dropout' : [0.0], 
        'u_transformer' : [UniformScaler(feature_range=(-1,1))],
        'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'a_transformer' : [UniformScaler(feature_range=(-1,1))],
        'dtype' : ['float32'],
        'dummy_transformer' : [DummyScaler()],
        'class_weight' : [[1.0, 1.0, 1.0]] #no_pinn and PINN_A
    }

    config = {}
    config.update(data_config)
    config.update(network_config)
    return config