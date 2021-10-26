from GravNN.Trajectories import RandomAsteroidDist, SurfaceDist
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Data import single_training_validation_split
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Visualization.PolyVisualization import PolyVisualization
import numpy as np
import os
import matplotlib.pyplot as plt
def main():
    poly_vis = PolyVisualization()
    planet = Eros()
    color = 'green'
    min_radius = 0

    # color='red'
    # min_radius = planet.radius*2
    
    max_radius = planet.radius*3

    directory = os.path.abspath('.') +"/Plots/Asteroid/" + str(np.round(min_radius,2)) + "_" +str(np.round(max_radius,2))+ "/"
    os.makedirs(directory, exist_ok=True)

    train_trajectory = RandomAsteroidDist(planet, [min_radius, max_radius], 20000, model_file=planet.obj_200k)
    surface_trajectory = SurfaceDist(planet, planet.obj_200k)
    surface_poly_gm = Polyhedral(planet, planet.obj_200k, trajectory=surface_trajectory).load(override=False)

    x_train, x_val = single_training_validation_split(train_trajectory.positions, N_train=2500, N_val=0) 
    # poly_vis.plot_polyhedron(surface_poly_gm.mesh, np.linalg.norm(np.zeros(np.shape(surface_poly_gm.accelerations)),axis=1),cbar=False)
    poly_vis.plot_polyhedron(surface_poly_gm.mesh, np.linalg.norm(surface_poly_gm.accelerations, axis=1),cbar=False, surface_colors=False)

    poly_vis.plot_position_data(x_train, alpha=0.5,color=color)
    plt.gca().set_xlim([-max_radius, max_radius])
    plt.gca().set_ylim([-max_radius, max_radius])
    plt.gca().set_zlim([-max_radius, max_radius])
    # poly_vis.save(plt.gcf(), directory+"Asteroid_Training.pdf")
    poly_vis.save(plt.gcf(), directory+"Asteroid_Training.jpeg")

if __name__ == "__main__":
    main()