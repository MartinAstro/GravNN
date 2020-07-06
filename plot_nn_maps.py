import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["RUNFILES_DIR"] = "/Users/johnmartin/Library/Python/3.7/share/plaidml"# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path
os.environ["PLAIDML_NATIVE_PATH"] = "/Users/johnmartin/Library/Python/3.7/lib/libplaidml.dylib" # libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path

from Visualization.Grid import Grid
from Visualization.MapVisualization import MapVisualization
from GravityModels.SphericalHarmonics import SphericalHarmonics
from CelestialBodies.Planets import Earth
from Trajectories.DHGridDist import DHGridDist
from GravityModels.NN_Base import NN_Base
from keras.utils.layer_utils import count_params

import pyshtools
import matplotlib.pyplot as plt
import numpy as np

from Trajectories.UniformDist import UniformDist
from Trajectories.RandomDist import RandomDist

from Trajectories.DHGridDist import DHGridDist
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics
from Preprocessors.MinMaxTransform import MinMaxTransform
from Support.transformations import sphere2cart, cart2sph, project_acceleration

from GravityModels.GravityModelBase import GravityModelBase

from GravityModels.NN_Base import NN_Base
from GravityModels.NNSupport.NN_DeepLayers import *
from GravityModels.NNSupport.NN_MultiLayers import *
from GravityModels.NNSupport.NN_SingleLayers import *

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta
import copy
from plot_sh_maps import phase_2
def round_stats(grid):
    avg = np.round(np.average(np.abs(grid)),2)
    med =  np.round(np.median(np.abs(grid)),2)
    return avg, med
def print_nn_params(nn, true_grid, grid):
    error_grid = copy.deepcopy(grid)
    error_grid -= true_grid
    error_grid = (error_grid/true_grid) * 100.0
    dr, dr_median = round_stats(error_grid.r)
    dtheta, dtheta_median = round_stats(error_grid.theta)
    dphi, dphi_median = round_stats(error_grid.phi)
    dtotal, dtotal_median = round_stats(error_grid.total)
    
    if not type(nn.opt) == type(Adadelta()):
        lr =  str(nn.lr)
    else:
        lr =  "N/A" 
    print(nn.model_func.__name__ + "\t&\t" + str(len(nn.x_train)) + "\t&\t" + str(nn.batch_size) + "\t&\t" + str(nn.opt.__class__.__name__) + "\t&\t" +  str(lr) + "\t&\t" + str(dr) + "\t&\t" + str(dtheta) + "\t&\t" + str(dphi) + "\t&\t" + str(dtotal) + "\\\\ \n" + str(dr_median) + "\t&\t" + str(dtheta_median) + "\t&\t" + str(dphi_median) + "\t&\t" + str(dtotal_median) + "\\\\")

    return dtotal_median

def main():
    # Initialize test data set
    density_deg = 175
    max_deg = 1000
    planet = Earth()
    sh_file = planet.sh_hf_file

    map_grid = DHGridDist(planet, planet.radius, degree=density_deg)

    # Loop through different NN configurations
    nn_list = []
    points = [1000, 10000]#, 100000]
    #points = [10000]
    for point_count in points:

        # Load the training data
        #trajectory = UniformDist(planet, planet.radius, point_count)
        trajectory = RandomDist(planet, [planet.radius, planet.radius*1.1], point_count)
        training_gravity_model = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
        training_gravity_model.load(override=False) 

        # Subtract off C20 from training data
        training_gravity_model_C20 = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory)
        training_gravity_model_C20.load(override=False) 
        training_gravity_model.accelerations -= training_gravity_model_C20.accelerations


        arch_list = [
                    Single_32_tanh,
                    Single_128_tanh,
                    Single_256_tanh,

                    Single_32_LReLU,
                    Single_128_LReLU,
                    Single_256_LReLU,
                    
                    Deep_8x3_LReLU,
                    Deep_16x3_LReLU,
                    Deep_32x3_LReLU,
                     #Deep_128x3_LReLU
                    ]

        
        batch_sizes = [10]#, 1]
        opt_list = [Adadelta, SGD]
        for arch in arch_list:
            for bs in batch_sizes:
                for opt in opt_list:
                    if opt.__name__ == SGD.__name__:
                        lr_list = [5E-2, 1E-2]
                    else:
                        lr_list = [0]
                    
                    for lr in lr_list:
                        if lr != 0:
                            opt_inst = opt(lr)
                        else:
                            opt_inst = opt()
                
                        preprocessor = MinMaxTransform([-1,1])
                        preprocessor.percentTest = 0.1
    
                        # Initialize NN
                        nn = NN_Base(trajectory, training_gravity_model, preprocessor)
                        nn.epochs = 1000
                        nn.batch_size = bs
                        nn.lr = lr
                        nn.opt = opt_inst
                        nn.patience = 20
                        #nn.loss = "mean_absolute_percentage_error"

                        nn.model_func = arch
                        nn.verbose = False
                        nn.forceNewNN = False
                        nn.trainNN()
                        nn.trajectory = map_grid
                        x_formatted = np.array(nn.x_train).reshape((len(nn.x_train),3))
                        nn.compute_percent_error(nn.model.predict(x_formatted), nn.y_train)
                        nn.plotMetrics()
                        #exit()
                        nn_list.append(nn)
                        print(nn.file_directory)



    # Generate Support Grids
    sh_all_gravityModel = SphericalHarmonics(sh_file, degree=max_deg, trajectory=map_grid)
    sh_C20_gravityModel = SphericalHarmonics(sh_file, degree=2, trajectory=map_grid)

    # Subtract of C22 from truth
    true_grid = Grid(gravityModel=sh_all_gravityModel)
    sh_20_grid = Grid(gravityModel=sh_C20_gravityModel)
    true_grid -= sh_20_grid

    # Plot NN Results
    map_viz = MapVisualization()
    grid_list = []
    nn_params = []
    median_error = []
    for i in range(len(nn_list)):
        grid = Grid(gravityModel=nn_list[i])
        grid_list.append(grid)
        fig, ax = map_viz.percent_maps(true_grid,grid, param="total", vlim=[0,100])
        #map_viz.save(fig_abs_err, str(i) + "_NN_Abs_Error.pdf")
        map_viz.save(fig, nn_list[i].file_directory+"NN_Rel_Error.pdf")
        median_err = print_nn_params(nn_list[i], true_grid, grid)
        median_error.append(median_err)
        nn_params.append(count_params(nn_list[i].model.trainable_weights))

    phase_2()
    plt.plot(nn_params, median_error)
    plt.savefig("Again.pdf")


    # fig2, ax2 = map_viz.component_error(grid_list, true_grid, points, "red", sh_20_grid) 
    # plt.xlabel(r"NN Training Points")
    # plt.title("NN Error w.r.t. C20")

    plt.show()





if __name__ == "__main__":
    main()
