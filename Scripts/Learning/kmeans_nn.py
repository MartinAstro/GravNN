from matplotlib import image
import numpy as np
import copy 
import tensorflow as tf
import tensorflow.keras as keras

from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.GravityModels.NNSupport.SupportFunc import plot_metrics
from sklearn.cluster import KMeans, DBSCAN

from random import seed
seed(0)

def calc_nodes_per_layer(n_clusters, layers, SH_coef):
    p = [layers-1,  6+n_clusters+layers, 3-SH_coef]
    nodes = int(np.floor(np.max(np.roots(p))))
    return nodes


map_vis = MapVisualization()
planet = Earth()
radius = planet.radius
model_file = planet.sh_hf_file
density_deg = 175
max_deg = 1000

trajectory = ReducedGridDist(planet, radius, degree=density_deg, reduction=0.25)
#trajectory = DHGridDist(planet, radius, degree=density_deg)

Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
Call_r0_grid = Grid(trajectory=trajectory, accelerations=Call_r0_gm.load())
C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
C20_r0_grid = Grid(trajectory=trajectory, accelerations=C20_r0_gm.load())
R0_pert_grid = Call_r0_grid - C20_r0_grid

input_vec = trajectory.positions
output_vec = R0_pert_grid.acceleration

input_vec = np.array(input_vec)
output_vec = np.array(output_vec)

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

map_scaler = StandardScaler()
map_encode = map_scaler.fit_transform(R0_pert_grid.total)

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

#%% Cluster the data
n_clusters = 5
train = np.transpose(np.vstack((input_train[:,0], output_train[:,0], )))
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(train)

# db = DBSCAN().fit(train)
# labels = db.labels_

#%% Encode the data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# %% (Optional train a NN to perform classification)
from GravNN.GravityModels.NNSupport.NN_hyperparam import NN_hyperparam, NN_hyperparam_classification

from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2

params = {}
params['kernel_initializer'] = 'glorot_normal'
params['kernel_regularizer'] = None # 'l2'
params['first_unit'] = 32
params['first_neuron'] = 32 
params['hidden_layers'] = 2
params['dropout'] = 0.0
params['batch_size'] = 10
params['lr'] = 0.001
params['epochs'] = 100
params['optimizer'] = Adam
params['shapes'] = 'brick'
params['losses'] = 'categorical_crossentropy'# 'binary_crossentropy'# 'mean_absolute_error'
params['activation'] = 'relu' #LeakyReLU()# 'relu' #

save_location = None
# hist, model = NN_hyperparam_classification(input_train, onehot_encoded, [], [],  params, verbose=1, save_location=save_location, validation_split=0.2)
# pred_labels = model.predict(input_train)
pred_labels = onehot_encoded

#%% Train the acceleration network with the outputs of the 
layers = 3
SH_coef = 100
nodes = calc_nodes_per_layer(n_clusters, layers, SH_coef=SH_coef)
print("Nodes: " + str(nodes))
params = {}
params['kernel_initializer'] = 'glorot_normal'
params['kernel_regularizer'] = 'l2' # None # 'l2'
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

input_w_labels = np.hstack((input_train, pred_labels, ))

save_location = None
hist, model = NN_hyperparam(input_w_labels, output_train, [], [],  params, verbose=1, save_location=save_location, validation_split=0.2)


# %% Get Stats
import matplotlib.pyplot as plt
pred = model.predict(input_w_labels)
pred_decode = copy.copy(pred)

pred_decode[:,0] = acc_r_scalar.inverse_transform(pred[:,0].reshape(len(pred), 1))[:,0]
pred_decode[:,1] = acc_theta_scalar.inverse_transform(pred[:,1].reshape(len(pred), 1))[:,0]
pred_decode[:,2] = acc_phi_scalar.inverse_transform(pred[:,2].reshape(len(pred), 1))[:,0]

print("True Max: " + str(output_vec.max()))
print("True Min: " + str(output_vec.min()))

print("Encode Max: " + str(pred.max()))
print("Encode Min: " + str(pred.min()))
print("Decode Max: " + str(pred_decode.max()))
print("Decode Min: " + str(pred_decode.min()))


# %% Plot Errors
params = model.count_params()

print(params)
coef = int(np.floor(np.max(np.roots([1, 1, -SH_coef]))))
Clm_r0_gm= SphericalHarmonics(model_file, degree=coef, trajectory=trajectory)
Clm_r0_grid = Grid(trajectory=trajectory, accelerations=Clm_r0_gm.load()
SH_error_grid = Clm_r0_grid - C20_r0_grid

SH_error_grid = np.sqrt(np.square(SH_error_grid.total - R0_pert_grid.total))
vlim = [0, np.max(SH_error_grid)]
fig_pert, ax = map_vis.plot_grid(SH_error_grid, "SH RSE [mGal]",vlim=vlim)
plt.title("SH Error")

pred_decode = np.linalg.norm(pred_decode,axis=1)
pred_grid = np.reshape(pred_decode, np.shape(R0_pert_grid.total))
fig_pert, ax = map_vis.plot_grid(pred_grid, "Pred Acceleration [mGal]")
plt.title("NN Pred")

error_grid = np.sqrt(np.square(pred_grid - R0_pert_grid.total))
fig_pert, ax = map_vis.plot_grid(error_grid, "NN RSE [mGal]", vlim=vlim)
plt.title("NN Error")

plot_metrics(hist)

#%% Compute Error
print("SH Avg Error: " + str(np.average(SH_error_grid)))
print("NN Avg Error: " + str(np.average(error_grid)))
print("Improvement:" + str((np.average(SH_error_grid) - np.average(error_grid))/np.average(SH_error_grid)*100))

plt.show()



