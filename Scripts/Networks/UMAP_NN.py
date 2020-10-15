from matplotlib import image
import numpy as np
import copy 
import tensorflow as tf
import tensorflow.keras as keras

from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.StateObject import StateObject
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.GravityModels.NNSupport.SupportFunc import plot_metrics
from sklearn.cluster import KMeans, DBSCAN
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.Reduced3dDist import Reduced3dDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from random import seed
import umap.plot
import os
seed(0)

def calc_nodes_per_layer(latent, layers, SH_coef):
    #p = [layers-1,  6+n_clusters+layers, 3-SH_coef]
    p = [layers-1, layers+latent+3, 1-latent - SH_coef]
    nodes = int(np.floor(np.max(np.roots(p))))
    return nodes

vis = VisualizationBase(save_directory=os.path.splitext(__file__)[0]  + "/../../../Plots/" +os.path.splitext(os.path.basename(__file__))[0] + "/",halt_formatting=True)

planet = Earth()
radius = planet.radius
model_file = planet.sh_hf_file
density_deg = 175
max_deg = 1000

#trajectory_reduced = Reduced3dDist(planet, radiusBounds=[radius, radius*1.05], layers=5, degree=density_deg, reduction=0.25)
#trajectory_reduced = ReducedRandDist(planet, radiusBounds=[radius, radius*1.05], points=15488, degree=density_deg, reduction=0.25)

# Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_reduced)
# C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory_reduced)

# Call_r0_state_obj = StateObject(gravityModel=Call_r0_gm)
# C20_r0_state_obj = StateObject(gravityModel=C20_r0_gm)
# R0_pert_state_obj = Call_r0_state_obj - C20_r0_state_obj


trajectory_reduced = ReducedGridDist(planet, radius*1.05, degree=density_deg, reduction=0.25)
#trajectory_reduced = DHGridDist(planet, radius, degree=density_deg)

Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_reduced)
C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory_reduced)

Call_r0_state_obj = Grid(trajectory=trajectory_reduced, accelerations=Call_r0_gm.load())
C20_r0_state_obj = Grid(trajectory=trajectory_reduced, accelerations=C20_r0_gm.load())
R0_pert_state_obj = Call_r0_state_obj - C20_r0_state_obj


input_vec = trajectory_reduced.positions
output_vec = R0_pert_state_obj.acceleration

input_vec = np.array(input_vec)
output_vec = np.array(output_vec)

# %%
# Transform the data

val_range = (0,1)
r_scaler = MinMaxScaler()#(feature_range=val_range)
theta_scaler = MinMaxScaler()#(feature_range=val_range) 
phi_scaler = MinMaxScaler()#(feature_range=val_range) 

acc_r_scalar = StandardScaler()
acc_theta_scalar = StandardScaler()
acc_phi_scalar = StandardScaler()

r_train_encoded = r_scaler.fit_transform(input_vec[:,0].reshape(len(input_vec), 1))[:,0]
theta_train_encoded = theta_scaler.fit_transform(input_vec[:,1].reshape(len(input_vec), 1))[:,0]
phi_train_encoded = phi_scaler.fit_transform(input_vec[:,2].reshape(len(input_vec), 1))[:,0]

acc_r_train_encoded = acc_r_scalar.fit_transform(output_vec[:,0].reshape(len(output_vec), 1))[:,0]
acc_theta_train_encoded = acc_theta_scalar.fit_transform(output_vec[:,1].reshape(len(output_vec), 1))[:,0]
acc_phi_train_encoded = acc_phi_scalar.fit_transform(output_vec[:,2].reshape(len(output_vec), 1))[:,0]

input_train = np.stack((r_train_encoded, theta_train_encoded, phi_train_encoded), axis=1)
output_train = np.stack((acc_r_train_encoded, acc_theta_train_encoded, acc_phi_train_encoded), axis=1)

data = np.hstack((input_train, output_train))


#%% Resolve Latent Space
latent = 6
reducer = umap.UMAP(n_neighbors=7, min_dist=0.0,n_components=latent)
embedding = reducer.fit_transform(data)


# %% (Train NN)
from GravNN.GravityModels.NNSupport.NN_hyperparam import NN_hyperparam, NN_hyperparam_classification

from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2
layers = 3
SH_coef = 10000
nodes = calc_nodes_per_layer(latent, layers, SH_coef=SH_coef)
print("Nodes: " + str(nodes))
params = {}
params['kernel_initializer'] = 'glorot_normal'
params['kernel_regularizer'] = None # None # 'l2'
params['first_unit'] = nodes
params['first_neuron'] = nodes 
params['hidden_layers'] = layers - 1 # first layer is already defined. 
params['dropout'] = 0.2
params['batch_size'] = 10
params['lr'] = 0.001
params['epochs'] = 50
params['optimizer'] = Adam
params['shapes'] = 'brick'
params['losses'] = 'mean_squared_error'# 'mean_absolute_error'
params['activation'] = 'relu' #LeakyReLU()# 'relu' #

save_location = None
hist, model = NN_hyperparam(input_train, embedding, [], [],  params, verbose=1, save_location=save_location, validation_split=0.2)


# %% Get Stats
import matplotlib.pyplot as plt
pred = model.predict(input_train)
pred = reducer.inverse_transform(pred)
pred_acc = pred[:,3:]
pred_decode = copy.copy(pred_acc)

pred_decode[:,0] = acc_r_scalar.inverse_transform(pred_acc[:,0].reshape(len(pred), 1))[:,0]
pred_decode[:,1] = acc_theta_scalar.inverse_transform(pred_acc[:,1].reshape(len(pred), 1))[:,0]
pred_decode[:,2] = acc_phi_scalar.inverse_transform(pred_acc[:,2].reshape(len(pred), 1))[:,0]

print("True Max: " + str(output_vec.max()))
print("True Min: " + str(output_vec.min()))

print("Encode Max: " + str(pred.max()))
print("Encode Min: " + str(pred.min()))
print("Decode Max: " + str(pred_decode.max()))
print("Decode Min: " + str(pred_decode.min()))


# %% Plot Errors
params = model.count_params()

print(params)
map_vis = MapVisualization()
coef = int(np.floor(np.max(np.roots([1, 1, -SH_coef]))))
Clm_r0_gm= SphericalHarmonics(model_file, degree=coef, trajectory=trajectory_reduced)
Clm_r0_state_obj = Grid(trajectory=trajectory_reduced, accelerations=Clm_r0_gm.load()
SH_error_state_obj = Clm_r0_state_obj - C20_r0_state_obj

SH_error_state_obj = np.sqrt(np.square(SH_error_state_obj.total - R0_pert_state_obj.total))
vlim = [0, np.max(SH_error_state_obj)]
fig_pert, ax = map_vis.plot_grid(SH_error_state_obj, "SH RSE [mGal]",vlim=vlim)
plt.title("SH Error")

pred_decode = np.linalg.norm(pred_decode,axis=1)
pred_state_obj = np.reshape(pred_decode, np.shape(R0_pert_state_obj.total))
fig_pert, ax = map_vis.plot_grid(pred_state_obj, "Pred Acceleration [mGal]")
plt.title("NN Pred")

error_state_obj = np.sqrt(np.square(pred_state_obj - R0_pert_state_obj.total))
fig_pert, ax = map_vis.plot_grid(error_state_obj, "NN RSE [mGal]", vlim=vlim)
plt.title("NN Error")

plot_metrics(hist)

#%% Compute Error
print("SH Avg Error: " + str(np.average(SH_error_state_obj)))
print("NN Avg Error: " + str(np.average(error_state_obj)))
print("Improvement:" + str((np.average(SH_error_state_obj) - np.average(error_state_obj))/np.average(SH_error_state_obj)*100))

plt.show()



