from GravNN.CelestialBodies.Asteroids import Toutatis
from GravNN.Trajectories import RandomAsteroidDist
from GravNN.Preprocessors import DummyScaler, UniformScaler
from sklearn.preprocessing import MinMaxScaler
def get_prototype_toutatis_config():
    data_config = {
        'planet' : [Toutatis()],
        'grav_file' : [Toutatis().model_lf],
        'distribution' : [RandomAsteroidDist],
        'N_dist' : [10000],
        'N_train' : [2500], 
        'N_val' : [150],
        'radius_min' : [0],
        'radius_max' : [Toutatis().radius + 5000.0],
        'acc_noise' : [0.00],
        'basis' : [None],
        'mixed_precision' : [False],
        'dtype' :['float32'],
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
