import os

os.environ["PATH"] += (
    os.pathsep
    + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"
)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Trajectories import DHGridDist
from GravNN.Visualization.MapBase import MapBase

np.random.seed(1234)
tf.random.set_seed(0)

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# * gca().get_lines()[n].get_xydata() lets you get the data from a curve


def plot_maps(config, model, map_trajectories):
    x_transformer = config["x_transformer"][0]
    a_transformer = config["a_transformer"][0]
    for name, map_traj in map_trajectories.items():
        model_file = map_traj.celestial_body.sh_file
        x, a, u = get_sh_data(
            map_traj,
            model_file,
            config["max_deg"][0],
            config["deg_removed"][0],
        )

        if config["basis"][0] == "spherical":
            x = cart2sph(x)
            a = project_acceleration(x, a)
            x[:, 1:3] = np.deg2rad(x[:, 1:3])

        x = x_transformer.transform(x)
        a = a_transformer.transform(a)

        U_pred, acc_pred = model.predict(x.astype("float32"))

        x = x_transformer.inverse_transform(x)
        a = a_transformer.inverse_transform(a)
        acc_pred = a_transformer.inverse_transform(acc_pred)

        if config["basis"][0] == "spherical":
            x[:, 1:3] = np.rad2deg(x[:, 1:3])
            a = invert_projection(x, a)
            invert_projection(
                x,
                acc_pred.astype(float),
            )  # numba requires that the types are the same

        grid_true = Grid(trajectory=map_traj, accelerations=a)
        grid_pred = Grid(trajectory=map_traj, accelerations=acc_pred)

        mapUnit = "mGal"
        map_vis = MapBase(mapUnit)
        map_vis.tick_interval = [60, 60]
        # plt.rc('text', usetex=False)
        map_vis.fig_size = map_vis.half_page
        map_vis.file_directory = (
            os.path.abspath(".") + "/Plots/OneOff/" + str(config["id"][0]) + "/"
        )

        vlim = [0, np.max(grid_pred.total) * 10000]
        fig_true, ax = map_vis.plot_grid(grid_true.total, "[mGal]", vlim=vlim)
        map_vis.save(fig_true, name + "/" + "true.pdf")
        plt.close()

        fig_pred, ax = map_vis.plot_grid(grid_pred.total, "[mGal]", vlim=vlim)
        map_vis.save(fig_pred, name + "/" + "pred.pdf")
        plt.close()


def main():
    planet = Earth()
    df_file = "Data/Dataframes/N_1000000_exp_norm_study.data"

    df = pd.read_pickle(df_file).sort_values(by="params", ascending=False)[:3]
    ids = df["id"].values

    for id_value in ids:
        tf.keras.backend.clear_session()

        config, model = load_config_and_model(df, model_id_file)

        density_deg = 180
        test_trajectories = {
            "Brillouin": DHGridDist(planet, planet.radius, degree=density_deg),
            "LEO": DHGridDist(planet, planet.radius + 420000.0, degree=density_deg),
            # "GEO" : DHGridDist(planet, planet.radius+35786000.0, degree=density_deg)
        }

        # plot standard metrics (loss, maps) the model
        plot_maps(config, model, test_trajectories)

        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
