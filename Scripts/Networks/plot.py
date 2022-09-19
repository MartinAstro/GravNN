
import os

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
#import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.Networks.Model import PINNGravityModel, load_config_and_model

from GravNN.Networks.Plotting import Plotting
from GravNN.Trajectories import DHGridDist

np.random.seed(1234)
tf.random.set_seed(0)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# * gca().get_lines()[n].get_xydata() lets you get the data from a curve
def main():
    planet = Earth()
    df_file = 'Data/Dataframes/temp_spherical.data'
    df_file = 'Data/Dataframes/new_pinn_constraints.data'
    df_file = 'Data/Dataframes/new_temp.data'
    df_file = 'Data/Dataframes/new_temp_small.data'
    df_file = 'Data/Dataframes/new_temp_long.data'

    df_file = 'Data/Dataframes/moon_test.data'

    df = pd.read_pickle(df_file)#[3:]#.sort_values(by='params')[2:]##[:2]
    ids = df['id'].values

    for id_value in ids:
        tf.keras.backend.clear_session()

        model_id = id_value
        config, model = load_config_and_model(model_id, df_file)

        # model = None
        # config = utils.get_df_row(model_id, df_file)


        plotter = Plotting(model, config)
        planet = config['planet'][0]
        density_deg = 180
        test_trajectories = {
            "Brillouin" : DHGridDist(planet, planet.radius, degree=density_deg),
            #"LEO" : DHGridDist(planet, planet.radius+420000.0, degree=density_deg),
            #"GEO" : DHGridDist(planet, planet.radius+35786000.0, degree=density_deg)
        }

        # plot standard metrics (loss, maps) the model
        #plotter.plot_maps(test_trajectories)
        plotter.plot_loss(log=True)

        # plot optional metrics (altitude plot)
        #plotter.plot_alt_curve('rse_median')
        #plotter.plot_data_alt_curve('rse_median')
        #plotter.plot_model_graph()
        
    plt.show()
    plt.close()






if __name__ == '__main__':
    main()
