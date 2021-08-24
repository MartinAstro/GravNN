import os
import copy

import numpy as np
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
        "override" : [False]
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
    laplace_train = np.zeros((1,))
    curl_train = np.zeros((3,))

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
    x_bar = transformers['x'].transform(x)
    y_bar = transformers['a'].transform(y)
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


def regress_nn(config, sampling_interval, sub_directory=None):
    directory = "/Users/johnmartin/Documents/GraduateSchool/Research/ML_Gravity/GravNN/Files/GravityModels/Regressed/" + sub_directory + "/"
    os.makedirs(directory, exist_ok=True)
    
    planet = Eros()
    model_file = planet.model_potatok
    remove_point_mass = False

    discard_data = True
    epochs = 100

    max_altitude = 10000.0

    x_max = planet.radius + max_altitude
    x_min = -x_max 

    a_max = planet.mu/planet.radius**2
    a_min = -a_max

    u_max = planet.mu/planet.radius
    u_min = planet.mu/(planet.radius + max_altitude)

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

    total_samples = 0
    # For each orbit, train the network
    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        x, a, u = get_poly_data(
            trajectory, model_file, remove_point_mass=[remove_point_mass]
        )
        for i in range(len(x)):
            x_inst_bar, y_inst_bar = preprocess_data(x[i], a[i], transformers, config)
            x_train.append(x_inst_bar)
            y_train.append(y_inst_bar)

        # Update the neural network with a batch of data
        regressor.update(np.array(x_train), np.array(y_train), iterations=epochs)

        total_samples += len(x_train)

        # optionally dump old data
        if discard_data: 
            x_train = []
            y_train = []

        try:
            time = trajectory.times[0]
        except:
            time = None 

        file_name = "%s/%s/PINN_%d_%d_%d_%d.data" % (
            planet.__class__.__name__,
            trajectory.__class__.__name__,
            regressor.model.config["num_units"][0],
            total_samples,
            time, 
            sampling_interval
            )
        regressor.model.config['PINN_constraint_fcn'] = [regressor.model.config['PINN_constraint_fcn'][0]]
        os.makedirs(os.path.dirname(directory+file_name),exist_ok=True)
        regressor.model.save(directory + file_name)
        pbar.update(k)



def main():

    params = {'pinn_constraint_fcn' : ['pinn_a']}
    config = get_hparams(params)
    regress_nn(config, sampling_interval=10*60, sub_directory='pinn_a')


    # params = {'pinn_constraint_fcn' : ['pinn_alc']}
    # config = get_hparams(params)
    # regress_nn(config, sampling_interval=10*60, sub_directory='pinn_alc')


    # params = {'network_type' : ['sph_pines_transformer'],
    #           'transformer_units' : [20]}
    # config = get_hparams(params)
    # regress_nn(config, sampling_interval=10*60, sub_directory='transformer_pinn_a')


    # params = {'network_type' : ['sph_pines_transformer'],
    #           'pinn_constraint_fcn' : ['pinn_alc'],
    #           'transformer_units' : [20]}
    # config = get_hparams(params)
    # regress_nn(config, sampling_interval=10*60, sub_directory='transformer_pinn_alc')

if __name__ == "__main__":
    main()
