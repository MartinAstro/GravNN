
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Trajectories import EphemerisDist, SurfaceDist
from GravNN.Visualization.PolyVisualization import PolyVisualization
from GravNN.CelestialBodies.Asteroids import Eros, Bennu
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Trajectories.utils import generate_orex_orbit_trajectories
import numpy as np
from mpl_toolkits import mplot3d
import os
def main():
    directory = os.path.abspath('.') +"/Plots/Asteroid/Trajectories/"
    os.makedirs(directory, exist_ok=True)

    # # Page 17 of 43 in "ORex Flight Dynamics and Navigation Design" B. Williams 2018 -- Space Sci Rev. has the following intervals:
    # start_date = 'August 17, 2018'

    # intervals = [94, # approach
    #              20, # preliminary survey
    #              31, # Orbital A
    #              63, # Detailed Survey 
    #              60, # Orbit B
    #              98, #Reconnaissance
    #              42, #TAG Rehearsal
    #              23, #Sample Collection
    #             ]

    # #website intervals
    # intervals = [108, # approach to Prelim
    #              28, # Prelim to A
    #              59, # Orbital A to Detailed Survey
    #              # 56, # Survey to Equatorial Stations
    #              107, #Survey to Orbit B
    #              51, # Orbit B to Orbit C
    #              57, # Orbit C to Recon A
    #              183, #Recon to rehearsal
    orbits = generate_orex_orbit_trajectories(60*10)

    # get et values one and two, we could vectorize str2et
    vis = VisualizationBase()
    # poly_vis = PolyVisualization()
    # surf_trajectory = SurfaceDist(Bennu(), Bennu().model_potatok)
    # x, a, u = get_poly_data(surf_trajectory, Bennu().model_potatok)
    # tri = Poly3DCollection(surf_trajectory.mesh.triangles)

    # https://www.jhuapl.edu/Content/techdigest/pdf/V23-N01/23-01-Holdridge.pdf -- NEAR Orbits
    for i in range(0, len(orbits)-1):
        fig = vis.new3DFig()
        #poly_vis.plot_polyhedron(surf_trajectory.mesh, np.linalg.norm(a,axis=1))
        positions = orbits[i].positions
        ax = plt.gca()
        ax.plot3D(positions[:,0], positions[:,1], positions[:,2], 'gray')
        
        ax.axes.set_xlim3d(left=np.min(positions[:,0]), right=np.max(positions[:,0])) 
        ax.axes.set_ylim3d(bottom=np.min(positions[:,1]), top=np.max(positions[:,1])) 
        ax.axes.set_zlim3d(bottom=np.min([np.min(positions[:,2]), -Bennu().radius]), top=np.max([np.max(positions[:,2]), Bennu().radius])) 
        vis.save(plt.gcf(), directory + "OREX_Orbit_"+str(i)+ ".pdf")
    #plt.show()

if __name__ == '__main__':
    main()