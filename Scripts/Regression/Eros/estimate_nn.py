from GravNN.Support.transformations import cart2sph
import os
import copy

import numpy as np
import matplotlib.pyplot as plt
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Networks.Configs import *
from GravNN.Preprocessors.UniformScaler import UniformScaler
from GravNN.Regression.NN import NN
from GravNN.Support.ProgressBar import ProgressBar
from GravNN.Trajectories import RandomAsteroidDist, EphemerisDist
from GravNN.Trajectories.utils import generate_near_orbit_trajectories
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

from GravNN.Networks.utils import (
    check_config_combos,
    configure_run_args,
    configure_tensorflow,
    load_hparams_to_config,
    set_mixed_precision,
)


def get_hparams(params={}):
    config = get_default_eros_config()

    hparams = {
        "learning_rate": [0.001*2],
        "batch_size": [131072 // 2],

        "PINN_constraint_fcn": ["pinn_a"],
        "num_units": [20],
        "beta" : [0.9],

        'schedule_type' : ['plateau'],
        "patience" : [250],
        "decay_rate" : [0.9],
        "min_delta" : [0.0001],
        "min_lr" : [0.0001],

        "lr_anneal" : [False],
        "remove_point_mass" : [False], # remove point mass from polyhedral model
        "override" : [False],
        'skip_normalization' : [False]
    }
    hparams.update(params)
    
    # Necessary to permuatate any combinations 
    args = configure_run_args(config, hparams)
    config = args[0][0]
    hparams = args[0][1]

    config = copy.deepcopy(config)
    config = load_hparams_to_config(hparams, config)

    check_config_combos(config)
    print(config)
    return config

def augment_y_data(a_train, config):
    pinn_constraint_fcn = config['PINN_constraint_fcn'][0].__name__
    laplace_train = np.zeros((len(a_train),1))
    curl_train = np.zeros((len(a_train),3))

    # TODO: Modify which variables are added to the state to speed up training and minimize memory footprint. 
    if pinn_constraint_fcn == "no_pinn":
        y_train = np.hstack([a_train]) 
    # Just use acceleration
    elif pinn_constraint_fcn == "pinn_A":
        y_train = np.hstack([a_train]) 
    
    # Accelerations + 2nd order constraints
    elif pinn_constraint_fcn == "pinn_AL":
        y_train = np.hstack([a_train, laplace_train]) 
    elif pinn_constraint_fcn == "pinn_ALC":
        y_train = np.hstack([a_train, laplace_train, curl_train]) 
    else:
        exit("No PINN Constraint Selected!")
    return y_train

def preprocess_data(x,y, transformers, config):
    x_bar = transformers['x'].transform(np.array(x))
    y_bar = transformers['a'].transform(np.array(y))
    y_bar = augment_y_data(y_bar, config)
    return x_bar, y_bar

def fit_transformers(x_dumb, a_dumb, u_dumb, config):
    
    x_dumb = np.array(x_dumb)
    a_dumb = np.array(a_dumb)
    u_dumb = np.array(u_dumb)

    # Preprocessing
    x_transformer = config['x_transformer'][0]
    a_transformer = config['a_transformer'][0]
    u_transformer = config['u_transformer'][0]
    a_bar_transformer = copy.deepcopy(config['a_transformer'][0])

    '''
    x_bar = (x-x0)/xs
    u_bar = (u-u0)/us

    The acceleration is a byproduct of the potential so 

    a_bar = du_bar/dx_bar

    but this equates to

    a_bar = d((u-u0)/us)/dx_bar = d(u/us)/dx_bar = 1/us*du/dx_bar = 1/us *du/(d((x-x0)/xs)) = 1/us * du/d(x/xs) = xs/us * du/dx = xs/us * a

    such that a_bar exists in [-1E2, 1E2] rather than [-1,1]
    '''

    # Designed to make position, acceleration, and potential all exist between [-1,1]
    x_dumb = x_transformer.fit_transform(x_dumb)
    u_dumb = u_transformer.fit_transform(np.repeat(u_dumb,3,axis=1))[:,0].reshape((-1,1))

    # Scale the acceleration by the potential 
    a_bar = a_transformer.fit_transform(a_dumb, scaler=1/(x_transformer.scale_/u_transformer.scale_)) # this is a_bar
    a_bar_transformer.fit(a_bar)
    a_dumb = a_bar # this should be on the order of [-100, 100]
    config['a_bar_transformer'] = [a_bar_transformer]

    transformers = {"x" : x_transformer,
                    "a" : a_transformer,
                    "u" : u_transformer,
                    "a_bar" : a_bar_transformer}

    config["x_transformer"] = [transformers["x"]]
    config["u_transformer"] = [transformers["u"]]
    config["a_transformer"] = [transformers["a"]]
    config["a_bar_transformer"] = [transformers["a_bar"]]
    
    return transformers

def get_replay_buffer(x_train, y_train, buffer_size):
    if len(x_train) < buffer_size:
        return x_train, y_train
    else:
        # sample data that is temporally closer
        # preference data that is at closer altitude
        x_train_sph = cart2sph(np.array(x_train))
        x_train_r = x_train_sph[:,0]
        prob_r = np.abs(1-(x_train_r-np.min(x_train_r))/(np.max(x_train_r)-np.min(x_train_r)))
        x_train_t = np.arange(len(prob_r))
        prob_t = x_train_t/ np.max(x_train_t)

        prob_sample = prob_r*prob_t
        idx = prob_sample.argsort()[-buffer_size:][::-1]
        print("Oldest Index = %d / %d" % (np.min(idx), len(x_train)))
        return np.array(x_train)[idx,:], np.array(y_train)[idx,:]


def get_replay_buffer_random(x_train, y_train, buffer_size, recency=3000):
    if len(x_train) < buffer_size:
        return x_train, y_train
    else:
        idx = np.arange(len(x_train))[::-1][recency:]
        idx = np.concatenate(idx, np.random.choice(len(x_train) - recency, size=buffer_size, replace=False))
        return np.array(x_train)[idx,:], np.array(y_train)[idx,:]

def regress_nn(config, sampling_interval, replay_buffer=None):  
    print(config['PINN_constraint_fcn'][0])
    planet = Eros()
    model_file = planet.obj_200k
    remove_point_mass = False

    discard_data = True
    epochs = 1000

    max_radius = planet.radius*3
    min_radius = planet.radius  # Brill radius - some extra room

    x_max = max_radius
    x_min = -x_max 

    a_max = planet.mu/min_radius**2
    a_min = -a_max

    u_max = planet.mu/(min_radius)
    u_min = planet.mu/max_radius

    x_extrema = [[x_min, x_max, 0.0]]
    a_extrema = [[a_min, a_max, 0.0]]
    u_extrema = [[u_min, u_max, 0.0]]

    transformers = fit_transformers(x_extrema, a_extrema, u_extrema, config)

    # Initialize the regressor
    regressor = NN(config)

    x_train = []
    y_train = []

    trajectories = generate_near_orbit_trajectories(sampling_inteval=sampling_interval)
    pbar = ProgressBar(len(trajectories), enable=True)

    plt.figure()
    total_samples = 0
    # For each orbit, train the network
    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        x, a, u = get_poly_data(
            trajectory, model_file, remove_point_mass=[remove_point_mass]
        )

        try:
            for i in range(len(x)):
                x_train.append(x[i])
                y_train.append(a[i])
        except:
            x_train = np.concatenate((x_train, x))
            y_train = np.concatenate((y_train, a))

            
        if replay_buffer is not None:
            x_train_sample, y_train_sample = get_replay_buffer(x_train, y_train, replay_buffer)
        else:
            x_train_sample = x_train
            y_train_sample = y_train

        x_train_sample, y_train_sample = preprocess_data(x_train_sample, y_train_sample, transformers, config)

        # Update the neural network with a batch of data
        regressor.update(np.array(x_train_sample), np.array(y_train_sample), iterations=epochs)

        plt.plot(regressor.model.history.history['loss'], label=str(k))
        # plt.show()
        if replay_buffer is not None or discard_data == False:
            total_samples = len(x_train)
        else:
            total_samples += len(x_train)

        # optionally dump old data
        if discard_data: 
            x_train = []
            y_train = []

        file_name = "%s/%s/%s_%s_%d.data" % (
            planet.__class__.__name__,
            trajectory.__class__.__name__,

            config['PINN_constraint_fcn'][0].__name__,
            str(replay_buffer), 
            total_samples,
            )
        regressor.model.config['PINN_constraint_fcn'] = [regressor.model.config['PINN_constraint_fcn'][0]]
        directory = os.path.curdir + "/GravNN/Files/GravityModels/Regressed/" 
        os.makedirs(os.path.dirname(directory+file_name),exist_ok=True)
        regressor.model.save(directory + file_name)
        pbar.update(k)
        # if k % 5 == 0:
        #     plt.legend()
        #     plt.show()
        #     plt.figure()
    plt.legend()
    plt.show()


def main():



    params = {'PINN_constraint_fcn' : ['pinn_a']}
    config = get_hparams(params)
    regress_nn(config, sampling_interval=10*60, replay_buffer=None)

    # params = {'PINN_constraint_fcn' : ['pinn_a']}
    # config = get_hparams(params)
    # regress_nn(config, sampling_interval=10*60, replay_buffer=5000)
    
    # params = {'PINN_constraint_fcn' : ['pinn_alc']}
    # config = get_hparams(params)
    # regress_nn(config, sampling_interval=10*60, replay_buffer=None)

    # params = {'PINN_constraint_fcn' : ['pinn_alc']}
    # config = get_hparams(params)
    # regress_nn(config, sampling_interval=10*60,  replay_buffer=5000)


if __name__ == "__main__":
    main()
