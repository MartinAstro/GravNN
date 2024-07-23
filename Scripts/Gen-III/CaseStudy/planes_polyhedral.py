import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer


def main():
    df = pd.read_pickle("Data/Dataframes/pinn_primary_figure_III.data")
    model_id = df["id"].values[-1]
    print(model_id)

    config, model = load_config_and_model(df, model_id)
    config["gravity_data_fcn"] = [get_hetero_poly_data]

    planet = config["planet"][0]
    radius_bounds = [-planet.radius * 3, planet.radius * 3]
    max_percent = 10

    model = Polyhedral(planet, planet.obj_200k)

    planes_exp = PlanesExperiment(
        model,
        config,
        radius_bounds,
        200,
    )
    planes_exp.load_model_data(model)
    planes_exp.run()
    vis_hetero = PlanesVisualizer(planes_exp)
    vis_hetero.plot(z_max=max_percent)
    vis_hetero.save(plt.gcf(), "Eros_Planes_homo")
    plt.show()


if __name__ == "__main__":
    main()
