        
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import SurfaceDist
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Visualization.DataVisSuite import DataVisSuite

def make_fcn_name_latex_compatable(name):
    components = name.split("_")
    return components[0] + r"$_{" + components[1] + r"}$"

def get_spherical_data(x, a):
    x_sph = cart2sph(x)
    a_sph = project_acceleration(x_sph, np.array(a, dtype=float))
    return x_sph, a_sph

def main():
    directory = os.path.abspath('.') +"/Plots/Asteroid/"
    os.makedirs(directory, exist_ok=True)

    data_vis = DataVisSuite(halt_formatting=False)
    data_vis.fig_size = data_vis.full_page

    planet = Eros()
    test_trajectory = SurfaceDist(planet, obj_file=planet.model_potatok)        
    test_poly_gm = Polyhedral(planet, planet.model_potatok, trajectory=test_trajectory).load(override=False)

    u_list, u_pred_list = [], []
    a_list, a_pred_list = [], []
    label_list = []
    df = pd.read_pickle('Data/Dataframes/useless_070621_v4.data')
    model_ids = df['id'].values[:]
    directory = (
            os.path.abspath(".")
            + "/Plots/Asteroid/"
        )
    for model_id in model_ids:
        config, model = load_config_and_model(model_id, df)
        extra_samples = config.get('extra_N_train', [None])[0]

        label = make_fcn_name_latex_compatable(config["PINN_constraint_fcn"][0].__name__)
        os.makedirs(directory, exist_ok=True)

        x = test_poly_gm.positions
        a = test_poly_gm.accelerations
        u = test_poly_gm.potentials
  
        data_pred = model.generate_nn_data(x)
        a_pred = data_pred['a']
        u_pred = data_pred['u']

        x_sph, a_sph = get_spherical_data(x, a)
        x_sph, a_sph_pred = get_spherical_data(x, a_pred)
        
        a_list.append(a_sph[:,0])
        u_list.append(u)
        
        a_pred_list.append(a_sph_pred[:,0])
        u_pred_list.append(u_pred[:,0])
        label_list.append(label)

    fig = data_vis.plot_box_and_whisker(u_list, u_pred_list, label=label_list)
    # fig.show()
    fig.write_image(directory+"Potential_Surface_Box.png", format='png')
    
    fig = data_vis.plot_box_and_whisker(a_list, a_pred_list, label=label_list)
    # fig.show()
    fig.write_image(directory+"Acceleration_Surface_Box.png", format='png')

    plt.show()

if __name__ == "__main__":
    main()
