from matplotlib import image
import numpy as np
import copy 

from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist

map_vis = MapVisualization()
planet = Earth()
radius = planet.radius
model_file = planet.sh_hf_file
density_deg = 175
max_deg = 1000

trajectory_reduced = ReducedGridDist(planet, radius, degree=density_deg, reduction=0.25)
Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_reduced)
Call_r0_grid = Grid(trajectory=trajectory_reduced, accelerations=Call_r0_gm.load())
C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory_reduced)
C20_r0_grid = Grid(trajectory=trajectory, accelerations=C20_r0_gm.load())
R0_pert_grid = Call_r0_grid - C20_r0_grid

input_vec = trajectory_reduced.positions
output_vec = R0_pert_grid.acceleration

input_vec = np.array(input_vec)
output_vec = np.array(output_vec)

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

val_range = (0,1)
r_scaler = MinMaxScaler()#(feature_range=val_range)
theta_scaler = MinMaxScaler()#(feature_range=val_range) 
phi_scaler = MinMaxScaler()#(feature_range=val_range) 

#z_scaler = StandardScaler() 
acc_r_scalar = StandardScaler()
acc_theta_scalar = StandardScaler()
acc_phi_scalar = StandardScaler()


pos_train, pos_test, acc_train, acc_test = train_test_split(input_vec, output_vec, test_size=0.3, random_state=43)

r_train_encoded = r_scaler.fit_transform(pos_train[:,0].reshape(len(pos_train), 1))[:,0]
theta_train_encoded= theta_scaler.fit_transform(pos_train[:,1].reshape(len(pos_train), 1))[:,0]
phi_train_encoded= phi_scaler.fit_transform(pos_train[:,2].reshape(len(pos_train), 1))[:,0]

r_test_encoded = r_scaler.transform(pos_test[:,0].reshape(len(pos_test), 1))[:,0]
theta_test_encoded = theta_scaler.transform(pos_test[:,1].reshape(len(pos_test), 1))[:,0]
phi_test_encoded = phi_scaler.transform(pos_test[:,2].reshape(len(pos_test), 1))[:,0]

acc_r_train_encoded = acc_r_scalar.fit_transform(acc_train[:,0].reshape(len(acc_train), 1))[:,0]
acc_theta_train_encoded = acc_theta_scalar.fit_transform(acc_train[:,1].reshape(len(acc_train), 1))[:,0]
acc_phi_train_encoded = acc_phi_scalar.fit_transform(acc_train[:,2].reshape(len(acc_train), 1))[:,0]

acc_r_test_encoded = acc_r_scalar.transform(acc_test[:,0].reshape(len(acc_test), 1))[:,0]
acc_theta_test_encoded = acc_theta_scalar.transform(acc_test[:,1].reshape(len(acc_test), 1))[:,0]
acc_phi_test_encoded = acc_phi_scalar.transform(acc_test[:,2].reshape(len(acc_test), 1))[:,0]

input_train = np.stack((r_train_encoded, theta_train_encoded, phi_train_encoded), axis=1)
output_train = np.stack((acc_r_train_encoded, acc_theta_train_encoded, acc_phi_train_encoded), axis=1)

input_test = np.stack((r_test_encoded, theta_test_encoded, phi_test_encoded), axis=1)
output_test = np.stack((acc_r_test_encoded, acc_theta_test_encoded, acc_phi_test_encoded), axis=1)


# %%
from GravNN.GravityModels.NNSupport.NN_hyperparam import NN_hyperparam
from GravNN.GravityModels.NNSupport.NN_RBF import NN_RBF

from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2

params = {}
nodes = 1024
params['epochs'] = 50
params['batch_size'] = 1
params['optimizer'] = Adadelta
params['kernel_initializer'] = GlorotUniform
params['kernel_regularizer'] = None
params['first_unit'] = nodes
params['first_neuron'] =  nodes #128
params['hidden_layers'] = 1
params['dropout'] = 0.0
params['lr'] = 0.005
params['activation'] = 'relu'
params['losses'] = 'mean_squared_error'# 'mean_squared_error'
params['beta'] = 0.5

save_location = None
hist, model = NN_RBF(input_train, output_train, input_test, output_test, params, verbose=1, save_location=save_location, validation_split=0.2)


# %%
import matplotlib.pyplot as plt
input_vec_encode = input_vec
input_vec_encode[:,0] = r_scaler.transform(input_vec_encode[:,0].reshape(len(input_vec_encode), 1))[:,0]
input_vec_encode[:,1] = theta_scaler.transform(input_vec_encode[:,1].reshape(len(input_vec_encode), 1))[:,0]
input_vec_encode[:,2] = theta_scaler.transform(input_vec_encode[:,2].reshape(len(input_vec_encode), 1))[:,0]

pred = model.predict(input_vec_encode)
pred_decode = copy.deepcopy(pred)
pred_decode[:,0] = acc_r_scalar.inverse_transform(pred[:,0].reshape(len(pred), 1))[:,0]
pred_decode[:,1] = acc_theta_scalar.inverse_transform(pred[:,1].reshape(len(pred), 1))[:,0]
pred_decode[:,2] = acc_phi_scalar.inverse_transform(pred[:,2].reshape(len(pred), 1))[:,0]


print("True Max: " + str(output_vec.max()))
print("True Min: " + str(output_vec.min()))

print("Encode Max: " + str(pred.max()))
print("Encode Min: " + str(pred.min()))
print("Decode Max: " + str(pred_decode.max()))
print("Decode Min: " + str(pred_decode.min()))

pred_decode = np.linalg.norm(pred_decode,axis=1)
R0_pert_grid.total = np.reshape(pred_decode, np.shape(R0_pert_grid.total))
fig_pert, ax = map_vis.plot_grid(R0_pert_grid.total, "Acceleration [mGal]")
plt.show()

