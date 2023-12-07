import matplotlib.pyplot as plt

from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Configs import get_default_eros_config
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer


def main():
    planet = Eros()
    model = Polyhedral(planet, planet.obj_200k)
    config = get_default_eros_config()
    config["obj_file"] = [Eros().obj_200k]

    planet = config["planet"][0]
    # config["gravity_data_fcn"] = [get_poly_data]
    R_multiplier = 5
    radius_bounds = [-planet.radius * R_multiplier, planet.radius * R_multiplier]

    planes_exp = PlanesExperiment(model, config, radius_bounds, 200)
    planes_exp.load_model_data(model)
    planes_exp.run()
    vis = PlanesVisualizer(
        planes_exp,
        # save_directory=os.path.abspath(".") + "/Plots/Eros/",
    )

    # vis.plot(z_max=100, log=True, annotate_stats=True)
    vis.plot(z_max=10, log=False, annotate_stats=True)

    # vis.save(plt.gcf(), "Eros_Planes.pdf")

    plt.show()


if __name__ == "__main__":
    main()
