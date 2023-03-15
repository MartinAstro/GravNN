import pandas as pd
import numpy as np
import os
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.CelestialBodies.Asteroids import Eros

import matplotlib.pyplot as plt
from GravNN.Networks.Model import load_config_and_model
from GravNN.GravityModels.Polyhedral import get_poly_data

def SRP(A2M, Cr, P_srp, r_body_2_sun):
    a_SRP = -Cr*A2M*P_srp/r_body_2_sun**2
    return a_SRP


def main():


    df = pd.read_pickle("Data/Dataframes/eros_point_mass_gen_III.data")

    model_id = df["id"].values[-1] 

    print(model_id)
    config, model = load_config_and_model(model_id, df)
    config['grav_file'] = [Eros().obj_8k]

    planet = config['planet'][0]
    config['gravity_data_fcn'] = [get_poly_data]
    radius_bounds = [-planet.radius*3, planet.radius*3]
    max_percent = 10


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
    # vis.save(plt.gcf(), "Eros_Planes.pdf")

    plt.show()



if __name__ == "__main__":
    main()