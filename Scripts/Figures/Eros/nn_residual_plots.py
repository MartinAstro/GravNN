        
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import RandomAsteroidDist
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Visualization.DataVisSuite import DataVisSuite
from GravNN.Networks.Data import get_raw_data

def make_fcn_name_latex_compatable(name):
    components = name.split("_")
    return components[0] + r"$_{" + components[1] + r"}$"

def get_spherical_data(x, a):
    x_sph = cart2sph(x)
    a_sph = project_acceleration(x_sph, np.array(a, dtype=float))
    return x_sph, a_sph

def overlay_hist(x_sph_train,twinx=True):
    if twinx:
        plt.gca().twinx()
    plt.hist(x_sph_train[:,0], bins=30,alpha=0.7,  range=[np.min(x_sph_train[:,0]), np.max(x_sph_train[:,0])], color='gray')#edgecolor='black', linewidth=0.5, fill=True)
    plt.ylabel("Frequency")

def plot_moving_average(x, y, y_pred, percent=True, color='black'):
    if not percent:
        diff = y - y_pred
    else:
        diff = np.abs(np.linalg.norm(y - y_pred,axis=1)/np.linalg.norm(y,axis=1))*100.0
    df = pd.DataFrame(data=diff, index=x)
    df.sort_index(inplace=True)
    rolling_avg = df.rolling(500, 100).mean()
    plt.plot(df.index, rolling_avg, c=color)       


def main():

    df = pd.read_pickle("Data/Dataframes/eros_residual_r.data")
    sh_regress_files = ['0R_3R/N_4_Noise_0.00.csv', 
                        '0R_3R/N_4_Noise_0.20.csv',
                        '0R_3R/N_16_Noise_0.00.csv', 
                        '0R_3R/N_16_Noise_0.20.csv']

    # df = pd.read_pickle("Data/Dataframes/eros_residual_r_bar.data")
    # sh_regress_files = ['2R_3R_Plus/N_4_Noise_0.00.csv', 
    #                     '2R_3R_Plus/N_4_Noise_0.20.csv',
    #                     '2R_3R_Plus/N_16_Noise_0.00.csv', 
    #                     '2R_3R_Plus/N_16_Noise_0.20.csv']

    # df = pd.read_pickle("Data/Dataframes/eros_residual_r_star.data")
    # sh_regress_files = ['2R_3R/N_4_Noise_0.00.csv', 
    #                     '2R_3R/N_4_Noise_0.20.csv',
    #                     '2R_3R/N_16_Noise_0.00.csv', 
    #                     '2R_3R/N_16_Noise_0.20.csv']

    data_vis = DataVisSuite(halt_formatting=False)
    data_vis.fig_size = data_vis.tri_vert_page

    planet = Eros()
    model_file = planet.obj_200k
    
    test_trajectory = RandomAsteroidDist(planet, [0, planet.radius*3], 50000, grav_file=[model_file])        
    test_poly_gm = Polyhedral(planet, model_file, trajectory=test_trajectory).load()
    
    x = test_poly_gm.positions
    a = test_poly_gm.accelerations
    x_sph, a_sph = get_spherical_data(x, a)

    config, model = load_config_and_model(df['id'].values[0], df)       
    x_train, a_train, _, _, _, _ = get_raw_data(config)
    x_sph_train, _ = get_spherical_data(x_train, a_train)


    data_vis.newFig()

    # Plot training distribution
    overlay_hist(x_sph_train,twinx=False)
    plt.grid(False)
    plt.gca().twinx()
    plt.grid(True, which='both')
    plt.gca().set_axisbelow(True)
    # Plot PINN Error
    marker_list = ['v', '.', 's']
    color_list = ['black', 'gray']
    for i in range(len(df['id'].values)):
        model_id = df['id'].values[i]
        config, model = load_config_and_model(model_id, df)
  
        a_pred = model.generate_acceleration(x)
        x_sph, a_sph_pred = get_spherical_data(x, a_pred)

        label = "PINN %s" % str(config['acc_noise'][0])
        data_vis.plot_residuals(x_sph[:,0], a_sph, a_sph_pred, 
                                alpha=0.5,label=label, 
                                ylabel='Acceleration')
        plot_moving_average(x_sph[:,0], a_sph, a_sph_pred, color=color_list[i])

    # Plot SH Error
    marker_list = ['v', '.', 's']
    color_list = ['magenta', 'yellow', 'cyan', 'pink']
    for i in range(len(sh_regress_files)):
        sh_file = 'GravNN/Files/GravityModels/Regressed/Eros/Residual/' + sh_regress_files[i]
        degree = int(os.path.basename(sh_file).split("N_")[1].split("_")[0])
        regressed_gm = SphericalHarmonics(sh_file, degree, trajectory=test_trajectory).load(override=True)

        a_pred = regressed_gm.accelerations
        x_sph, a_sph_pred = get_spherical_data(x, a_pred)

        acc_noise = float(os.path.basename(sh_file).split("_")[-1].split(".csv")[0])
        label = "SH %d %.1f" % (degree, acc_noise)
        data_vis.plot_residuals(x_sph[:,0], a_sph, a_sph_pred, 
                                alpha=0.5,label=label, 
                                ylabel='Acceleration')
        plot_moving_average(x_sph[:,0], a_sph, a_sph_pred, color=color_list[i])

    plt.gca().set_yscale('log')
    plt.gca().set_ylim([1E-2, 5E2])
    plt.gca().set_xlim([-1000, 51000])
    plt.legend(loc='lower left')

    data_vis.plot_radii(x_sph[:,0], 
                        vlines=[planet.radius, config['radius_min'][0], config['radius_max'][0]], 
                        vline_labels=[r'$r_{Brill}$', r'$r_{min}$', r'$r_{max}$'])


    directory = os.path.abspath(".") + "/Plots/Asteroid/"
    os.makedirs(directory, exist_ok=True)

    file_name = "%s_%s_%s_Residual.png" %(
        str(int(np.round(config["radius_min"][0], 2))),
        str(int(np.round(config["radius_max"][0], 2))),
        str(config.get('extra_N_train', [None])[0]))
    
    data_vis.save(plt.gcf(), directory+file_name)
    plt.show()

    # Snippets attempting to get the scatter plots above te histogram layer

    # plt.gca().yaxis.set_ticks_position('right')
    # plt.gca().yaxis.set_label_position('right')

    # plt.gca().yaxis.set_ticks_position('left')
    # plt.gca().yaxis.set_label_position('left')

    # plt.gcf().axes[0].set_zorder(1)
    # plt.gcf().axes[1].set_zorder(0)



if __name__ == "__main__":
    main()
