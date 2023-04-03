import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Planets import Earth
from GravNN.Networks.Configs import get_default_earth_config
from GravNN.Networks.Data import DataSet
from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Support.transformations_tf import cart2sph, compute_projection_matrix

plt.rc("text", usetex=True)


def get_data_config(max_degree, deg_removed, max_radius):
    config = get_default_earth_config()
    config.update(
        {
            "radius_max": [max_radius],
            "N_dist": [10000],
            "N_train": [9500],
            "N_val": [500],
            "max_deg": [max_degree],
            "deg_removed": [deg_removed],
            "dummy_transformer": [DummyScaler()],
        },
    )
    return config


def plot_acceleration_components(r, a):
    min = a.min()
    max = a.max()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.scatter(r, a[:, 0], alpha=0.5, s=2, label="$a_1$")
    plt.ylim([min, max])
    plt.grid()
    plt.subplot(1, 3, 2)
    plt.scatter(r, a[:, 1], alpha=0.5, s=2, label="$a_2$")
    plt.ylim([min, max])
    plt.grid()
    plt.subplot(1, 3, 3)
    plt.scatter(r, a[:, 2], alpha=0.5, s=2, label="$a_3$")
    plt.ylim([min, max])
    plt.grid()


def main():
    # spherical harmonic model
    planet = Earth()
    max_degree = 1000
    degree_removed = 2

    config = get_data_config(max_degree, degree_removed, max_radius=planet.radius * 5)
    data = DataSet(data_config=config)

    R = planet.radius
    r_mag = np.linalg.norm(data.raw_data["x_train"], axis=1)
    r_mag_norm = r_mag / R
    a_train = data.raw_data["a_train"].squeeze()

    plot_acceleration_components(r_mag_norm, a_train)

    # x_B = tf.matmul(BN, tf.reshape(x, (-1,3,1)))
    # this will give ~[1, 1E-8, 1E-8]

    x_sph = cart2sph(data.raw_data["x_train"].squeeze())
    BN = compute_projection_matrix(x_sph)
    a_sph = np.reshape(a_train, (-1, 3, 1))
    a_sph_B = np.matmul(BN, a_sph).squeeze()
    plot_acceleration_components(r_mag_norm, a_sph_B)

    plt.show()


if __name__ == "__main__":
    main()
