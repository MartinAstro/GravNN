import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Support.transformations import cart2sph
from GravNN.Trajectories import *
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)


def main():
    trajectories = []
    names = []
    planet = Earth()
    points = 1000000
    radius_bounds = [planet.radius, planet.radius + 420000.0]

    # Exponential Brillouin
    invert = False
    scale_parameter = 420000.0 / 3.0
    traj = ExponentialDist.ExponentialDist(
        planet,
        radius_bounds,
        points,
        scale_parameter=[scale_parameter],
        invert=[invert],
    )
    trajectories.append(traj)
    names.append("Exp 1")

    scale_parameter = 420000.0 / 10.0
    traj = ExponentialDist.ExponentialDist(
        planet,
        radius_bounds,
        points,
        scale_parameter=[scale_parameter],
        invert=[invert],
    )
    trajectories.append(traj)
    names.append("Exp 2")

    # Exponential LEO
    invert = True
    scale_parameter = 420000.0 / 3.0
    traj = ExponentialDist.ExponentialDist(
        planet,
        radius_bounds,
        points,
        scale_parameter=[scale_parameter],
        invert=[invert],
    )
    trajectories.append(traj)
    names.append("Exp Prime 1")

    scale_parameter = 420000.0 / 10.0
    traj = ExponentialDist.ExponentialDist(
        planet,
        radius_bounds,
        points,
        scale_parameter=[scale_parameter],
        invert=[invert],
    )
    trajectories.append(traj)
    names.append("Exp Prime 2")

    # Gaussian
    radius_bounds = [planet.radius, np.inf]
    mu = planet.radius + 420000.0
    sigma = 420000.0 / 3.0
    traj = GaussianDist.GaussianDist(
        planet,
        radius_bounds,
        points,
        mu=[mu],
        sigma=[sigma],
    )
    trajectories.append(traj)
    names.append("Normal 1")

    sigma = 420000.0 / 10.0
    traj = GaussianDist.GaussianDist(
        planet,
        radius_bounds,
        points,
        mu=[mu],
        sigma=[sigma],
    )
    trajectories.append(traj)
    names.append("Normal 2")

    # # Random Brillouin
    # RandomDist.RandomDist()

    # Plot composite curve
    vis = VisualizationBase()
    fig, ax = vis.newFig()

    for i in range(len(trajectories)):
        positions = cart2sph(trajectories[i].positions)
        plt.hist(positions[:, 0], bins=100, alpha=0.5, label=names[i])

    plt.xlabel("Altitude [km]")
    plt.ylabel("Frequency")
    plt.xlim([planet.radius, planet.radius + 420000 * 2.0])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
