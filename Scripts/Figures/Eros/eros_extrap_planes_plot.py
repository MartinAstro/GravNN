import pandas as pd
import numpy as np
import os
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.CelestialBodies.Asteroids import Eros

import matplotlib.pyplot as plt
from GravNN.Networks.Model import load_config_and_model

class CustomPlanesVisualizer(PlanesVisualizer):
    def __init__(self, exp, **kwargs):
        super().__init__(exp, **kwargs)
        plt.rc('font', size=7)

    def plot(self, percent_max=100):
        self.max = percent_max
        self.newFig()
        
        x = self.experiment.x_test
        y = self.experiment.percent_error_acc
        cbar_label = "$\mathbf{a}$ \% Error"
        cbar_label = None
        self.plot_plane(x,y, plane='xy', colorbar_label=cbar_label)

        plt.tight_layout()

def main():

    df = pd.read_pickle("Data/Dataframes/eros_PINN_III_extrapolation_v3.data")
    model_id = df["id"].values[-1] # PINN I
    config, model = load_config_and_model(model_id, df)
    config['grav_file'] = [Eros().obj_8k]

    planet = config['planet'][0]
    max_radius = planet.radius*5
    
    radius_bounds = [-max_radius, max_radius]
    max_percent = 10

    planes_exp = PlanesExperiment(model, config, radius_bounds, 30)
    planes_exp.run()
    vis = CustomPlanesVisualizer(planes_exp)
    vis.fig_size = vis.half_page_default
    vis.plot(percent_max=max_percent)
    plt.savefig("Plots/PINNII/Eros_Planes.pdf")  

    

    model_id = df["id"].values[-2] # PINN III
    config, model = load_config_and_model(model_id, df)
    config['grav_file'] = [Eros().obj_8k]
    config['fuse_models'] = [False]

    planes_exp = PlanesExperiment(model, config, radius_bounds, 30)
    planes_exp.run()
    vis = CustomPlanesVisualizer(planes_exp)
    vis.fig_size = vis.half_page_default
    vis.plot(percent_max=max_percent)
    plt.savefig("Plots/PINNIII/Eros_Planes.pdf")  

    plt.show()



if __name__ == "__main__":
    main()