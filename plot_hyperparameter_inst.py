import os, sys, keras, pickle, talos
import numpy as np
from numpy.random import seed

from Trajectories.UniformDist import UniformDist
from Trajectories.DHGridDist import DHGridDist
from Trajectories.RandomDist import RandomDist
from CelestialBodies.Planets import Earth
from Preprocessors.MinMaxTransform import MinMaxTransform
from Preprocessors.StandardTransform import StandardTransform
from Support.transformations import sphere2cart, cart2sph, project_acceleration, check_fix_radial_precision_errors
from GravityModels.NNSupport.NN_hyperparam import NN_hyperparam
from GravityModels.NN_Base import NN_Base
from GravityModels.SphericalHarmonics import SphericalHarmonics
from Visualization.Grid import Grid
from Visualization.MapVisualization import MapVisualization
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adadelta, Nadam, Adam, RMSprop

from talos.utils.recover_best_model import recover_best_model
from talos import Analyze, Evaluate, Scan, Predict, Deploy, Restore
from plot_nn_maps import print_nn_params
from keras.utils.layer_utils import count_params
import matplotlib.pyplot as plt

seed(1)

# Load a prior model if interested. 
# r = Restore('./hyper_uniform_minmax_full.zip') # Uniform Min Max 3
# a = Analyze('./Hyperparams/Uniform/070820172255.csv')

r = Restore('./hyper_uniform_minmax_v2_0.zip') # Uniform Min Max v2.0
a = Analyze('./Hyperparams/Uniform/070920073557.csv')

count_params(r.model.trainable_weights)
weights = r.model.trainable_weights# include BIASES so there are way more than we realize!
df =a.data

# # Generate training data to scale the preprocessor 
planet = Earth()
point_count = 10000
radBounds = planet.radius
trajectory = UniformDist(planet, planet.radius, point_count)
# radBounds = [planet.radius, planet.radius*1.1] 
# trajectory = RandomDist(planet, radBounds, point_count)

gravityModel = SphericalHarmonics(planet.sh_file, degree=None, trajectory=trajectory)
gravityModel.load() 
gravityModelC20 = SphericalHarmonics(planet.sh_file, degree=2, trajectory=trajectory)
gravityModelC20.load()

pos_sphere = cart2sph(trajectory.positions)
pos_sphere = check_fix_radial_precision_errors(pos_sphere)
acc_proj = project_acceleration(pos_sphere, gravityModel.accelerations)
acc_projC20 = project_acceleration(pos_sphere, gravityModelC20.accelerations)
acc_proj = acc_proj - acc_projC20

preprocessor = MinMaxTransform()
#preprocessor = StandardTransform()
preprocessor.percentTest = 0.00
preprocessor.split(pos_sphere, acc_proj)
preprocessor.fit()

# # Generate the test grid
density_deg = 175
map_grid = DHGridDist(planet, np.average(radBounds), degree=density_deg)

gravityModelMap = SphericalHarmonics(planet.sh_file, degree=None, trajectory=map_grid)
gravityModelMap.load() 
gravityModelMapC20 = SphericalHarmonics(planet.sh_file, degree=2, trajectory=map_grid)
gravityModelMapC20.load() 

test_pos_sphere = cart2sph(map_grid.positions)
test_pos_sphere = check_fix_radial_precision_errors(test_pos_sphere)
test_acc_proj = project_acceleration(test_pos_sphere, gravityModelMap.accelerations)
test_acc_projC20 = project_acceleration(test_pos_sphere, gravityModelMapC20.accelerations)
test_acc_proj = test_acc_proj - test_acc_projC20



x_train, y_train = preprocessor.apply_transform()
x_test, y_test = preprocessor.apply_transform(test_pos_sphere, test_acc_proj)


prediction = r.model.predict(x_train)
error_inst = abs(np.divide((prediction - y_train),y_train)*100)
print("TRAINING ENCODE")
print(np.average(error_inst))
print(np.median(error_inst))

prediction = r.model.predict(x_train)
x_decode, y_decode = preprocessor.invert_transform(x_train, prediction)
error_inst = abs(np.divide((y_decode - acc_proj),acc_proj)*100)
print("TRAINING DECODE")
print(np.average(error_inst))
print(np.median(error_inst))

prediction = r.model.predict(x_test)
x_decode, y_decode = preprocessor.invert_transform(x_test, prediction)
error_inst = abs(np.divide((y_decode - test_acc_proj),test_acc_proj)*100)
print("TESTING")
print(np.average(error_inst))
print(np.median(error_inst))

# Pick the best model and evaluate the average absolute difference
#e = Evaluate(t)
#error = e.evaluate(x_test,y_test, metric='val_accuracy', task='continuous', asc=False)

# # Generate Error Map


true_grid = Grid(gravityModel=gravityModelMap)
sh_20_grid = Grid(gravityModel=gravityModelMapC20)
true_grid -= sh_20_grid #these values are projected

#model = t.best_model('val_accuracy')
model = r.model
nn = NN_Base(model, preprocessor, test_traj=map_grid)

# Plot NN Results
map_viz = MapVisualization()
grid = Grid(gravityModel=nn, override=True)
fig, ax = map_viz.percent_maps(true_grid,grid, param="total", vlim=[0,100])
#map_viz.save(fig, nn.file_directory+"NN_Rel_Error.pdf")
median_err = print_nn_params(nn, true_grid, grid)
count_params(nn.model.trainable_weights)
print(median_err) 
plt.show()

