import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pyshtools
import pickle
import tensorflow as tf

from keras.optimizers import SGD, Adadelta
from keras.utils.layer_utils import count_params
from sklearn.model_selection import train_test_split
from talos import Analyze, Deploy, Evaluate, Predict, Restore, Scan
from talos.utils.recover_best_model import recover_best_model
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform , glorot_normal, glorot_uniform
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import model_from_json

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.GravityModels.NN_Base import NN_Base
from GravNN.GravityModels.NNSupport.NN_DeepLayers import *
from GravNN.GravityModels.NNSupport.NN_hyperparam import NN_hyperparam
from GravNN.GravityModels.NNSupport.NN_MultiLayers import *
from GravNN.GravityModels.NNSupport.NN_SingleLayers import *
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Preprocessors.MinMaxTransform import MinMaxTransform
from GravNN.Preprocessors.MinMaxStandardTransform import MinMaxStandardTransform

#from ..Training.run_hyperparameter_inst import compute_error
from GravNN.Support.transformations import (cart2sph,
                                     check_fix_radial_precision_errors,
                                     project_acceleration, sphere2cart)
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization

map_vis = MapVisualization(unit='mGal')
planet = Earth()
sh_file = planet.sh_hf_file
max_deg = 1000
density_deg = 175

trajectory_r0 = DHGridDist(planet, planet.radius, degree=density_deg)
Call_r0_gm = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory_r0)
Call_r0_grid = Grid(trajectory=trajectory_r0, accelerations=Call_r0_gm.load())
C20_r0_gm= SphericalHarmonics(sh_file, degree=2, trajectory=trajectory_r0)
C20_r0_grid = Grid(trajectory=trajectory_r0, accelerations=C20_r0_gm.load())
R0_pert_grid = Call_r0_grid - C20_r0_grid

trajectory_leo = DHGridDist(planet, planet.radius + 330*1000, degree=density_deg) 
Call_leo_gm = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory_leo)
Call_leo_grid = Grid(trajectory=trajectory_leo, accelerations=Call_leo_gm.load())
C20_leo_gm = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory_leo)
C20_leo_grid = Grid(trajectory=trajectory_leo, accelerations=C20_leo_gm.load())
LEO_pert_grid = Call_leo_grid - C20_leo_grid

def convert_fig(fig, ax):
        fontsize = 12
        ax.tick_params(labelsize=fontsize)
        cbar.set_label('Speedup', fontsize=fontSize)
        ax.set_ylabel("Local Size Z", fontsize=fontSize)
        ax.set_xlabel("Local Size X", fontsize=fontSize)
def fit_preprocessor(trajectory, preprocessor):
    """
    1. Compute the perturbations for the training trajectory (Uniform or Random)
    2. Divide that data into a train and validation set
    3. Preprocess that data / fit the transforms

    Args:
        trajectory (TrajectoryBase): [Data used to train the NN, typically Uniform or Random Distribution]
        preprocessor (PreprocessorBase): [Preprocessor used to transform the training data]
    """
    gravityModel = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    gravityModel.load() 
    gravityModelC20 = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory)
    gravityModelC20.load()

    pos_sphere = cart2sph(trajectory.positions)
    pos_sphere = check_fix_radial_precision_errors(pos_sphere)
    acc_proj = project_acceleration(pos_sphere, gravityModel.accelerations)
    acc_projC20 = project_acceleration(pos_sphere, gravityModelC20.accelerations)
    acc_proj = acc_proj - acc_projC20

    preprocessor.percentTest = 0.3
    preprocessor.split(pos_sphere, acc_proj)
    preprocessor.fit()

def get_train_test(Call_gm, C20_gm, preprocessor):
    """
    1. Take the test grid and compute its perturbations
    2. Apply the preprocessor to the test grid
    3. Return the training and testing data

    Args:
        Call_gm (GravityModelBase): [Full gravity model of the test set]
        C20_gm (GravityModelBase): [C20 gravity model of the test set]
        preprocessor (PreprocessorBase): [preprocessor used for training the original NN]

    Returns:
        [x_train, x_test, y_train, y_test]
    """
    test_pos_sphere = cart2sph(Call_gm.positions)
    test_pos_sphere = check_fix_radial_precision_errors(test_pos_sphere)
    test_acc_proj = project_acceleration(test_pos_sphere, Call_gm.accelerations)
    test_acc_projC20 = project_acceleration(test_pos_sphere, C20_gm.accelerations)
    test_acc_proj = test_acc_proj - test_acc_projC20

    x_train, x_val, y_train, y_val = preprocessor.apply_transform()
    x_test, y_test = preprocessor.apply_transform(test_pos_sphere, test_acc_proj)
    return x_train, x_test, y_train, y_test

def generate_se_map(trajectory, preprocessor, model, vlim, point_count,name):
    ####### Fit the preprocessor on the original training data
    fit_preprocessor(trajectory, preprocessor)

    ####### Generate the test data
    x_train, x_test, y_train, y_test = get_train_test(Call_r0_gm, C20_r0_gm, preprocessor)
    nn = NN_Base(model, preprocessor, test_traj=trajectory_r0)

    ####### Plot NN Results
    nn_grid = Grid(trajectory=trajectory_r0, accelerations=nn.load(), override=True)
    fig, ax = map_vis.plot_grid_rmse(nn_grid, R0_pert_grid, vlim=vlim)

    map_vis.save(fig, name + "_NN_RMSE.pdf")
    compute_error(model, x_train, y_train, x_test, y_test, preprocessor)
    print("Params: " + str(count_params(nn.model.trainable_weights)))
    plt.show()

def generate_LEO_se_map(trajectory, preprocessor, model, vlim, point_count,name):
    ####### Fit the preprocessor on the original training data
    fit_preprocessor(trajectory, preprocessor)

    ####### Generate the test data
    x_train, x_test, y_train, y_test = get_train_test(Call_leo_gm, C20_leo_gm, preprocessor)
    nn = NN_Base(model, preprocessor, test_traj=trajectory_leo)

    ####### Plot NN Results
    nn_grid = Grid(trajectory=trajectory_leo, accelerations=nn.load(), override=True)
    fig, ax = map_vis.plot_grid_rmse(nn_grid, LEO_pert_grid, vlim=vlim)

    map_vis.save(fig, name + "_NN_LEO_RMSE.pdf")
    compute_error(model, x_train, y_train, x_test, y_test, preprocessor)
    print("Params: " + str(count_params(nn.model.trainable_weights)))
    plt.show()

def plot_se_v_coef(trajectory, preprocessor, point_count, vlim=None):
    std = np.std(R0_pert_grid.total)
    mask = R0_pert_grid.total > 3*std

    ####### Fit the preprocessor to the original training data and pull out the correct 
    fit_preprocessor(trajectory, preprocessor)
    x_train, x_test, y_train, y_test = get_train_test(Call_r0_gm, C20_r0_gm, preprocessor)

    param_list = []
    nn_feat_rse_list = []
    nn_rse_list = []
    fig_loss, ax = map_vis.newFig()
    cases = [1, 2, 3, 4, 5, 6]
    training_handles = []
    validation_handles = []
    for i in cases:
        ####### Load the Final NN Weights and Generate Grid
        model, history = load_final_model(i, trajectory, preprocessor, point_count)

        nn = NN_Base(model, preprocessor, test_traj=trajectory_r0)
        nn_grid = Grid(trajectory=trajectory_r0, accelerations=nn.load(), override=True)

        ####### Plot the Loss for each NN 
        N = len(history['loss'])
        ax, = plt.plot(np.linspace(1,N, N), history['loss'], label='Loss ' + str(i))
        training_handles.append(ax)
        ax = plt.scatter(np.linspace(1,N, N), history['val_loss'], label='Val Loss ' + str(i), s=1)
        validation_handles.append(ax)
        plt.ylim(vlim)

        ####### Collect RSE 
        param_list.append(count_params(model.weights))
        nn_rse_list.append(np.average(np.sqrt(np.square(nn_grid.total - R0_pert_grid.total))))
        nn_feat_rse_list.append(np.average(np.sqrt(np.square((nn_grid.total - R0_pert_grid.total))),weights=mask))
    fontSize = 12
    ax = plt.gca()
    ax.tick_params(labelsize=fontSize)
    #cbar.set_label('Speedup', fontsize=fontSize)
    ax.set_ylabel("Loss", fontsize=fontSize)
    ax.set_xlabel("Epoch", fontsize=fontSize)
    training_legend = plt.legend(handles=training_handles, loc='upper left')

    validation_legend= plt.legend(handles=validation_handles, loc='upper right')
    ax = plt.gca().add_artist(training_legend)
   
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    map_vis.save(fig_loss, "NN_Losses.pdf")

    ####### Unpack RSE from SH
    with open('../GravNN/Files/Results/SH_RMSE.data', 'rb') as f:
        coefficient_list = pickle.load(f)
        rse_list = pickle.load(f)
        rse_feat_list = pickle.load(f)
    
    ####### Plot comparative RSE
    coefficient_list = np.array(coefficient_list)
    rse_list = np.array(rse_list)
    rse_feat_list = np.array(rse_feat_list)

    param_list = np.array(param_list)
    nn_rse_list = np.array(nn_rse_list)
    nn_feat_rse_list = np.array(nn_feat_rse_list)

    print(param_list)
    map_vis.newFig()
    plt.semilogy(coefficient_list, rse_list, label='SH Map', linestyle='-', color='b')
    plt.semilogy(coefficient_list, rse_feat_list, label='SH Features', linestyle='--', color='b')
    
    plt.xlabel("M Parameters")
    plt.ylabel("RMSE")
    map_vis.save(plt.gcf(), "SH_RMSE_2D.pdf")

    plt.semilogy(param_list[:3], nn_rse_list[:3], label='Single NN Map', linestyle='-', color='g')
    plt.semilogy(param_list[:3], nn_feat_rse_list[:3], label='Single NN Features', linestyle='--', color='g')

    plt.semilogy(param_list[3:], nn_rse_list[3:], label='Deep NN Map', linestyle='-', color='r')
    plt.semilogy(param_list[3:], nn_feat_rse_list[3:], label='Deep NN Features', linestyle='--', color='r')
    plt.legend()

    plt.xlabel("M Parameters")
    plt.ylabel("RMSE")
    map_vis.save(plt.gcf(), "NN_RMSE_2D.pdf")

def plot_LEO_se_v_coef(trajectory, preprocessor, point_count, vlim=None):
    std = np.std(LEO_pert_grid.total)
    mask = LEO_pert_grid.total > 3*std

    ####### Fit the preprocessor to the original training data and pull out the correct 
    fit_preprocessor(trajectory, preprocessor)
    x_train, x_test, y_train, y_test = get_train_test(Call_leo_gm, C20_leo_gm, preprocessor)

    param_list = []
    nn_feat_rse_list = []
    nn_rse_list = []
    fig_loss, ax = map_vis.newFig()
    cases = [1, 2, 3, 4, 5, 6]
    training_handles = []
    validation_handles=[]
    for i in cases:
        ####### Load the Final NN Weights and Generate Grid
        model, history = load_final_model(i, trajectory, preprocessor, point_count)

        nn = NN_Base(model, preprocessor, test_traj=trajectory_leo)
        nn_grid = Grid(trajectory=trajectory_leo, accelerations=nn.load(), override=True)

        ####### Plot the Loss for each NN 
        N = len(history['loss'])
        ax, = plt.plot(np.linspace(1,N, N), history['loss'], label='Loss ' + str(i))
        training_handles.append(ax)
        ax = plt.scatter(np.linspace(1,N, N), history['val_loss'], label='Val Loss ' + str(i), s=1)
        validation_handles.append(ax)
        plt.ylim(vlim)

        ####### Collect RSE 
        param_list.append(count_params(model.weights))
        nn_rse_list.append(np.average(np.sqrt(np.square(nn_grid.total - LEO_pert_grid.total))))
        nn_feat_rse_list.append(np.average(np.sqrt(np.square((nn_grid.total - LEO_pert_grid.total))),weights=mask))

    fontSize = 12
    ax = plt.gca()
    ax.tick_params(labelsize=fontSize)
    #cbar.set_label('Speedup', fontsize=fontSize)
    ax.set_ylabel("Loss", fontsize=fontSize)
    ax.set_xlabel("Epoch", fontsize=fontSize)
    training_legend = plt.legend(handles=training_handles, loc='upper left')

    validation_legend= plt.legend(handles=validation_handles, loc='upper right')
    ax = plt.gca().add_artist(training_legend)
   
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    map_vis.save(fig_loss, "NN_LEO_Losses.pdf")

    ####### Unpack RSE from SH
    with open('../GravNN/Files/Results/SH_LEO_RMSE.data', 'rb') as f:
        coefficient_list = pickle.load(f)
        rse_list = pickle.load(f)
        rse_feat_list = pickle.load(f)
    
    ####### Plot comparative RSE
    coefficient_list = np.array(coefficient_list)
    rse_list = np.array(rse_list)
    rse_feat_list = np.array(rse_feat_list)

    param_list = np.array(param_list)
    nn_rse_list = np.array(nn_rse_list)
    nn_feat_rse_list = np.array(nn_feat_rse_list)

    print(param_list)
    map_vis.newFig()
    plt.semilogy(coefficient_list, rse_list, label='SH Map', linestyle='-', color='b')
    plt.semilogy(coefficient_list, rse_feat_list, label='SH Features', linestyle='--', color='b')

    plt.xlabel("M Parameters")
    plt.ylabel("RMSE")
    map_vis.save(plt.gcf(), "SH_RMSE_LEO_2D.pdf")

    plt.semilogy(param_list[:3], nn_rse_list[:3], label='Single NN Map', linestyle='-', color='g')
    plt.semilogy(param_list[:3], nn_feat_rse_list[:3], label='Single NN Features', linestyle='--', color='g')

    plt.semilogy(param_list[3:], nn_rse_list[3:], label='Deep NN Map', linestyle='-', color='r')
    plt.semilogy(param_list[3:], nn_feat_rse_list[3:], label='Deep NN Features', linestyle='--', color='r')
    plt.legend()

    plt.xlabel("M Parameters")
    plt.ylabel("RMSE")
    map_vis.save(plt.gcf(), "NN_RMSE_LEO_2D.pdf")

def plot_nn_projection(trajectory, preprocessor, point_count, vlim=None):
    ####### Fit the preprocessor to the original training data and pull out the correct 
    fit_preprocessor(trajectory, preprocessor)

    cases = [1, 2, 3, 4, 5, 6]
    for i in cases:
        ####### Load the Final NN Weights and Generate Grid
        model, history = load_final_model(i, trajectory, preprocessor, point_count)

        nn = NN_Base(model, preprocessor, test_traj=trajectory_r0)
        nn_grid = Grid(trajectory=trajectory_r0, accelerations=nn.load(), override=True)
        fig,ax = map_vis.newFig()
        im = map_vis.new_map(nn_grid.total, vlim=vlim)
        map_vis.add_colorbar(im, "Case "+str(i))
        #fig, ax = map_vis.plot_grid(nn_grid.total, "CASE "+str(i))
        map_vis.save(plt.gcf(), "case" + str(i) + "_NN_Output.pdf")

def load_final_model(case, trajectory, preprocessor, point_count):
    # experiment_dir = trajectory.__class__.__name__ + "/" +    \
    #                                 preprocessor.__class__.__name__ + "/" + \
    #                                 str(point_count)
    experiment_dir = trajectory.trajectory_name + preprocessor.__class__.__name__
    experiment_dir =  experiment_dir.replace(', ', '_')
    experiment_dir =  experiment_dir.replace('[', '_')
    experiment_dir =  experiment_dir.replace(']', '_')

    save_location = "../GravNN/Files/Final_NN/" + experiment_dir + "/case_"+str(case) + "/"
    with open(save_location + "model.json", 'r') as f:
        json = f.read()
    model = model_from_json(json)
    model.load_weights(save_location + "model.h5")

    try:
        with open(save_location + "history.data", 'rb') as f:
            history = pickle.load(f)
    except:
        history = None
    return model, history 


def main():
    point_count = 259200 # 0.5 Deg
    planet = Earth()
    preprocessor = MinMaxStandardTransform()

    vlim = [0.0, 0.75]
    loss_vlim_r0 =  [0.55, 1.2] 
    loss_vlim_leo = [0, 1.0]

    #trajectory_train = UniformDist(planet, radius, point_count)
    trajectory_train = RandomDist(planet, [planet.radius , planet.radius + 5000], point_count) #R0
    trajectory_train_leo = RandomDist(planet, [planet.radius+330.0*1000-2500 , planet.radius + 330.0*1000+2500], point_count) #LEO

    # cases = [1, 2, 3, 4, 5, 6]
    # for case in cases:
    #     model, history = load_final_model(case, trajectory_train_leo, preprocessor, point_count)
    #     name = "case" + str(case)
    #     generate_LEO_se_map(trajectory_train_leo, preprocessor, model, vlim, point_count, name)

    # for case in cases:
    #     model, history = load_final_model(case, trajectory_train, preprocessor, point_count)
    #     name = "case" + str(case)
    #     vlim = [0, 10]
    #     generate_se_map(trajectory_train, preprocessor, model, vlim, point_count, name)

    ### PLOT NN LOSS FUNCTION AND ERROR 
    plot_se_v_coef(trajectory_train, preprocessor, point_count, loss_vlim_r0)
    plot_LEO_se_v_coef(trajectory_train_leo, preprocessor, point_count, loss_vlim_leo)

    ### PLOT NN PROJECTION
    #plot_nn_projection(trajectory_train, preprocessor, point_count, [0,10])
    #plot_nn_projection(trajectory_train_leo, preprocessor, point_count, [0,10])

    plt.show()
if __name__ == "__main__":
    main()
