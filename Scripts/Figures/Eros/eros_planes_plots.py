import pandas as pd
import numpy as np
import os
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.CelestialBodies.Asteroids import Eros

import matplotlib.pyplot as plt
from GravNN.Networks.Model import load_config_and_model

def SRP(A2M, Cr, P_srp, r_body_2_sun):
    a_SRP = -Cr*A2M*P_srp/r_body_2_sun**2
    return a_SRP


def main():
    # df = pd.read_pickle("Data/Dataframes/eros_pinn_III_040622.data")
    # model_id = df["id"].values[4] # ALC without extra
    # radius_bounds = [-planet.radius*500, planet.radius*500]
    # max_percent = 20

    # df = pd.read_pickle("Data/Dataframes/eros_residual_r_bar_trans_alc.data")
    # df = pd.read_pickle("Data/Dataframes/eros_pinn_II_III_warm_start.data")
    df = pd.read_pickle("Data/Dataframes/eros_comp_planes.data")
    
    model_id = df["id"].values[0] # ALC without extra
    model_id = df["id"].values[-1] # ALC without extra



    # df = pd.read_pickle("Data/Dataframes/eros_residual_r.data")

    print(model_id)
    config, model = load_config_and_model(model_id, df)
    config['grav_file'] = [Eros().obj_8k]

    planet = config['planet'][0]

    
    radius_bounds = [-planet.radius*1, planet.radius*1]
    max_percent = 1

    # planes_exp = PlanesExperiment(model, config, [-planet.radius*1, planet.radius*1], 30)
    planes_exp = PlanesExperiment(model, config, radius_bounds, 30)
    planes_exp.run()
    vis = PlanesVisualizer(planes_exp, save_directory=os.path.abspath(".")+"/Plots/Eros/")

    flux = 1367 # W/m^2
    c  = 3E8 # m/s
    Cr = 1.2
    A2M = 0.01 #m^2/kg
    semi_major_eros = 1.45 # AU

    P_srp_1_AU = flux/c
    P_srp_eros = P_srp_1_AU / (semi_major_eros/1)**2
    a_srp = P_srp_eros*A2M*Cr # m/s^2

    vis.set_SRP_contour(a_srp)

    vis.plot(percent_max=max_percent)
    vis.save(plt.gcf(), "Eros_Planes.pdf")
    # vis.plot_RMS()

    plt.show()



if __name__ == "__main__":
    main()