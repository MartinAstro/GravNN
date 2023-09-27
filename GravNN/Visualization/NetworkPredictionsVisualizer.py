import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.Analysis.DataInterface import DataInterface
from GravNN.Networks.Losses import *
from GravNN.Support.transformations import cart2sph, project_acceleration


class NetworkPredictionsVisualizer(DataInterface):
    def __init__(
        self,
        model,
        config,
        points,
        radius_bounds=None,
        random_seed=1234,
        remove_J2=False,
    ):
        super().__init__(
            model,
            config,
            points,
            radius_bounds=radius_bounds,
            random_seed=random_seed,
            remove_J2=remove_J2,
        )
        self.gather_data()

    def plot_accelerations(self):
        plt.figure()
        a_hat = np.linalg.norm(self.a_pred, axis=1)
        a_true = np.linalg.norm(self.accelerations, axis=1)

        r = np.linalg.norm(self.positions, axis=1)
        plt.scatter(r, a_hat, label="predicted")
        plt.scatter(r, a_true, label="true")
        plt.legend()

    def plot_potentials(self):
        plt.figure()
        r = np.linalg.norm(self.positions, axis=1)
        plt.scatter(r, self.u_pred, label="predicted")
        plt.scatter(r, self.potentials, label="true")
        plt.legend()

    def plot_acceleration_differences(self, sph=True):
        """
        Plots the difference between the predicted
        and true accelerations as histograms
        """
        da = self.accelerations - self.a_pred
        if sph:
            da = project_acceleration(cart2sph(self.positions), da)

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.hist(da[:, 0], 250)
        plt.subplot(3, 1, 2)
        plt.hist(da[:, 1], 250)
        plt.subplot(3, 1, 3)
        plt.hist(da[:, 2], 250)
        plt.xlabel("Acceleration")

    def plot_potential_differences(self):
        plt.figure()
        plt.hist(self.potentials - self.u_pred, 1000)
        plt.xlabel("Potentials")
        plt.show()

    def plot_u_nn(self):
        x = self.metric_exp.positions

        x_transformer = self.model.config["x_transformer"][0]
        self.model.config["u_transformer"][0]
        x = x_transformer.transform(x)
        # u_nn = self.model.network(x)
        outputs = self.model.network.layers[-2].output
        new_model = tf.keras.Model(inputs=self.model.network.input, outputs=outputs)
        u_nn = new_model(x)

        r = np.linalg.norm(x, axis=1)
        plt.figure()
        plt.scatter(r, u_nn)

    def plot(self):
        self.plot_accelerations()
        self.plot_potentials()
        self.plot_acceleration_differences()
        self.plot_potential_differences()

        self.plot_u_nn()


def main():
    from GravNN.Networks.Model import load_config_and_model

    df_file, idx = "Data/Dataframes/earth_high_alt4.data", -4
    df = pd.read_pickle(df_file)
    model_id = df.iloc[idx]["id"]
    config, model = load_config_and_model(df, model_id)
    exp = NetworkPredictionsVisualizer(model, config)
    exp.run()
    plt.show()


if __name__ == "__main__":
    main()
