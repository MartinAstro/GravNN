
import keras
import pickle
import talos
import numpy as np
from numpy.random import seed

from Trajectories.UniformDist import UniformDist
from Trajectories.DHGridDist import DHGridDist
from Trajectories.RandomDist import RandomDist
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics
from Preprocessors.MinMaxTransform import MinMaxTransform
from Preprocessors.RobustTransform import RobustTransform
from Preprocessors.StandardTransform import StandardTransform
from Preprocessors.MaxAbsTransform import MaxAbsTransform
from Visualization.Grid import Grid
from Visualization.MapVisualization import MapVisualization

import os, sys
sys.path.append(os.path.dirname(__file__) + "/build/PinesAlgorithm/")
import PinesAlgorithm

def pines_example():
    degree = 4
    gravityModel = SphericalHarmonics(os.path.dirname(os.path.realpath(__file__))  + "/../Files/GravityModels/GGM03S.txt", 10)
    planet = Earth(gravityModel)

    trajectory = UniformDist(planet, planet.radius, 100)

    positions = np.reshape(trajectory.positions, (len(trajectory.positions)*3))
    pines = PinesAlgorithm.PinesAlgorithm(planet.radius, planet.mu, degree)
    accelerations = pines.compute_acc(positions, gravityModel.C_lm, gravityModel.S_lm)
    accelerations = np.reshape(np.array(accelerations), (int(len(np.array(accelerations))/3), 3))
    print(accelerations)

def main():
    planet = Earth()

    point_count = 720*720*2
    #point_count = 259200 # 0.5 Deg
    #point_count = 64800 # 1 Deg
    #point_count = 10368 #2.5 Deg
    #point_count = 2592  # 5.0 Deg
    
    sh_file = planet.sh_hf_file
    max_deg = 1000
    #trajectory = UniformDist(planet, planet.radius, point_count)

    LEO_radius = planet.radius + 330.0*1000.0
    #trajectory = RandomDist(planet, [LEO_radius - 33.0*1000.0 , LEO_radius + 33.0*1000.01], point_count)
    trajectory = RandomDist(planet, [LEO_radius - 2.5*1000.0 , LEO_radius + 2.5*1000.0], point_count)

    
    #trajectory = RandomDist(planet, [planet.radius, planet.radius+5000], point_count) #5 kilometer band
    #trajectory = RandomDist(planet, [planet.radius, planet.radius+50000], point_count) #50 kilometer band



    gravityModelMap = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    gravityModelMap.load() 
    gravityModelMapC20 = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory)
    gravityModelMapC20.load() 


if __name__ == "__main__":
    main()