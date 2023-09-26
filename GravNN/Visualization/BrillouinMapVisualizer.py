import os

import matplotlib.pyplot as plt

from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.GravityModels.SphericalHarmonics import (
    SphericalHarmonics,
    SphericalHarmonicsDegRemoved,
)
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.Grid import Grid
from GravNN.Trajectories import DHGridDist
from GravNN.Visualization.MapBase import MapBase


class BrillouinMapVisualizer(MapBase):
    def __init__(self, planet, true_model, density_deg=180):
        super().__init__(unit="m/s^2")
        # configure grid at brillouin sphere
        DH_trajectory = DHGridDist(planet, planet.radius, degree=density_deg)
        self.trajectory = DH_trajectory
        self.tick_interval = [60, 60]

        # compute true accelerations
        self.true_model = true_model
        self.true_model.configure(DH_trajectory)
        self.true_model.load()

        self.grid_true = Grid(
            trajectory=DH_trajectory,
            accelerations=self.true_model.accelerations,
        )
        self.grid_pred = None

        # filename for saving plots is basename
        # extract class name
        self.basename = os.path.basename(__file__).split(".")[0]

        # BrillouinMapVisualizer_bennu
        self.file_name_base = f"{self.basename}_{planet.body_name}"

    def plot(self, value, **kwargs):
        kwargs.get("new_fig", True)
        kwargs.get("colorbar", True)
        kwargs.get("title", None)
        kwargs.get("label", None)

        # avg = round(float(np.average(value)), sigfigs=2)
        # std = 3 * np.std(value)

        self.plot_grid(
            value,
            "$m^2/s^2$",
            orientation="horizontal",
            loc="top",
            labels=False,
            ticks=False,
            # vlim=[clamp(avg - std, 0, np.inf), avg + std],
        )
        # plt.gcf().axes[0].annotate(
        #     "Avg: " + str(avg),
        #     xy=(0.05, 0.05),
        #     xycoords="axes fraction",
        #     fontsize="small",
        #     c="white",
        # )

    def generate_grid_predictions(self, model):
        try:
            # first try loading the model
            model.trajectory = self.trajectory
            model.load()
            acc_pred = model.accelerations
        except Exception:
            # if unavailable, compute manually
            acc_pred = model.compute_acceleration(self.trajectory.positions)
        return Grid(trajectory=self.trajectory, accelerations=acc_pred)

    def plot_true_accelerations(self, model):
        self.plot(
            self.grid_true.total,
            label="True Accelerations [$m^2/s^2$]",
        )
        self.save(plt.gcf(), f"{self.file_name_base}_true_{self.save_name}.pdf")

    def plot_pred_accelerations(self, model):
        grid_pred = self.generate_grid_predictions(model)
        self.plot(
            grid_pred.total,
            label="Predicted Accelerations [$m^2/s^2$]",
        )
        self.save(plt.gcf(), f"{self.file_name_base}_pred_{self.save_name}.pdf")

    def plot_acceleration_error(self, model):
        grid_pred = self.generate_grid_predictions(model)
        grid_error = (self.grid_true - grid_pred) / self.grid_true * 100
        self.plot(
            grid_error.total,
            label="Acceleration Errors [%]",
        )
        self.save(plt.gcf(), f"{self.file_name_base}_error_{self.save_name}.pdf")

    def plot_all(self, model, save_name=""):
        self.save_name = save_name
        self.plot_true_accelerations(model)
        self.plot_pred_accelerations(model)
        self.plot_acceleration_error(model)


def earth_maps():
    earth = Earth()
    earth_gm = SphericalHarmonics(earth.sh_file, degree=1000)
    df_file = "Data/Dataframes/earth_revisited_032723.data"
    _, earth_nn = load_config_and_model(df_file, idx=-1)

    vis = BrillouinMapVisualizer(earth, earth_gm)
    vis.plot_all(earth_nn, "PINN")

    # Make ground truth without C22
    earth_gm = SphericalHarmonicsDegRemoved(earth.sh_file, degree=1000, remove_deg=2)

    vis = BrillouinMapVisualizer(earth, earth_gm)
    earth_55 = SphericalHarmonicsDegRemoved(earth.sh_file, degree=55, remove_deg=2)
    vis.plot_all(earth_55, "SH_55")

    earth_110 = SphericalHarmonicsDegRemoved(earth.sh_file, degree=110, remove_deg=2)
    vis.plot_all(earth_110, "SH_110")


def bennu_maps():
    bennu = Bennu()
    bennu_gm = Polyhedral(bennu, bennu.obj_file)
    _, bennu_nn = load_config_and_model("Data/Dataframes/useless_070621_v4.data", idx=0)

    vis = BrillouinMapVisualizer(bennu, bennu_gm)
    vis.plot_all(bennu_nn, "PINN")


def main():
    earth_maps()
    # bennu_maps()


if __name__ == "__main__":
    main()
