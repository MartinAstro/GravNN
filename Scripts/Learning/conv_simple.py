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

from random import seed
seed(0)

map_vis = MapVisualization()
planet = Earth()
radius = planet.radius
model_file = planet.sh_hf_file
density_deg = 100
max_deg = 1000

trajectory_reduced = ReducedGridDist(planet, radius, degree=density_deg, reduction=0.25)
Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_reduced)
Call_r0_grid = Grid(gravityModel=Call_r0_gm)
C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory_reduced)
C20_r0_grid = Grid(gravityModel=C20_r0_gm)
R0_pert_grid = Call_r0_grid - C20_r0_grid

input_vec = trajectory_reduced.positions
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


# %%
from GravNN.GravityModels.NNSupport.NN_Conv import NN_Conv_Simple

from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2

params = {}
params['epochs'] = 10
params['batch_size'] = 100
params['kernel_initializer'] = GlorotUniform
params['kernel_regularizer'] = None
params['hidden_layers'] = 1
params['dropout'] = 0.0
params['lr'] = 0.005
params['activation'] = 'relu'
params['losses'] = 'mean_squared_error'# 'mean_squared_error'

save_location = None
hist, model = NN_Conv_Simple(map_encode, input_train, output_train, params, verbose=1, save_location=save_location)


# %%
import matplotlib.pyplot as plt
input_vec_encode = input_vec
input_vec_encode[:,0] = r_scaler.transform(input_vec_encode[:,0].reshape(len(input_vec_encode), 1))[:,0]
input_vec_encode[:,1] = theta_scaler.transform(input_vec_encode[:,1].reshape(len(input_vec_encode), 1))[:,0]
input_vec_encode[:,2] = theta_scaler.transform(input_vec_encode[:,2].reshape(len(input_vec_encode), 1))[:,0]

map_encode = np.full((len(input_train),np.shape(map_encode)[0],np.shape(map_encode)[1]), map_encode)
pred = model.predict([map_encode, input_train])

layer_outputs = [layer.output for layer in model.layers[:6]]
activation_model = keras.Model(inputs=model.input,
                                                            outputs=layer_outputs)
map_test = map_encode[0].reshape((1, map_encode[0].shape[0], map_encode[0].shape[1]))
acc_test = input_train[0].reshape((1, input_train[0].shape[0]))
activations = activation_model.predict([map_test, acc_test])
plot_layers = False
if plot_layers:
        
    plt.matshow(map_encode[0])
    plt.title("Original!")
    plt.matshow(activations[1][0,:,:,0])
    plt.title("Conv2d 1")

    print(activations[2].shape)
    plt.matshow(activations[2][0,:,:,0])
    plt.title("Pooling 1")

    plt.matshow(activations[3][0,:,:,0])
    plt.title("Conv2d 2")

    plt.matshow(activations[4][0,:,:,0])
    plt.title("Pooling 2")

    plt.matshow(activations[5][0,:,:,0])
    plt.title("Conv2d 3")

# for i in range(activations[2].shape[3]):
#     plt.matshow(activations[2][0,:,:,i])
#     plt.title(str(i))
plt.show()




# images = np.reshape(train_images[0:25], (-1, 28, 28, 1))
# tf.summary.image("Convolutional_Layer", images, max_outputs=1, step=0)

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

