import os

import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer


def main():
    df = pd.read_pickle("Data/Dataframes/heterogeneous_asymmetric_080523.data")
    model_id = df["id"].values[-1]
    print(model_id)

    config, model = load_config_and_model(df, model_id)
    config["gravity_data_fcn"] = [get_hetero_poly_data]

    planet = config["planet"][0]
    radius_bounds = [-planet.radius * 2, planet.radius * 2]
    max_percent = 10

    ###########################################
    # PINN Error
    ###########################################

    planes_exp = PlanesExperiment(
        model,
        config,
        radius_bounds,
        100,
        omit_train_data=True,
    )
    planes_exp.run()
    vis_hetero = PlanesVisualizer(
        planes_exp,
        save_directory=os.path.abspath(".") + "/Plots/Eros/",
    )
    vis_hetero.plot(z_max=max_percent)  # , annotate_stats=True)
    vis_hetero.save(plt.gcf(), "Eros_Planes_hetero.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
