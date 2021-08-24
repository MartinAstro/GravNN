
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Trajectories import EphemerisDist, SurfaceDist
from GravNN.Visualization.PolyVisualization import PolyVisualization
from GravNN.CelestialBodies.Asteroids import Eros
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
import numpy as np
import os
def main():
    directory = os.path.abspath('.') +"/Plots/Asteroid/Trajectories/"
    os.makedirs(directory, exist_ok=True)
    orbits = [
            # Approach
            'Aug 17, 2018', 

            # Preliminary Survey
            'Dec 03, 2018',

            # Orbital A
            'Dec 31, 2018',

            # Detailed Survey: Baseball Diamond
            'Feb 28, 2019',

            # Detailed Survey: Equatorial Stations
            'Apr 25, 2019',
            'May 02, 2019',
            'May 09, 2019',
            'May 16, 2019',
            'May 23, 2019',
            'May 30, 2019',
            'June 6, 2019',

            # Orbital B
            'Jun 15, 2019', # approx

            # Orbital C
            'Aug 05, 2019', # approx
            
            # Recon A
            'Oct 01, 2019',
            'Oct 31, 2019'

            # Recon B


            # Orbital R
            'Nov 01, 2019',
            'Jan 31, 2020',

            # Recon C


            # Rehearsal
            'Apr 01, 2020',
            'Aug 31, 2020',

            'Nov 03, 2019',
            'Dec 07, 2019',
            'Dec 13, 2019',
            'Jan 24, 2019',
            'Jan 28, 2019',
            'Jan 29, 2019',
            'Feb 02, 2019',
            'Feb 06, 2019']

    # get et values one and two, we could vectorize str2et
    vis = VisualizationBase()
    poly_vis = PolyVisualization()
    surf_trajectory = SurfaceDist(Eros(), Eros().model_potatok)
    x, a, u = get_poly_data(surf_trajectory, Eros().model_potatok)
    tri = Poly3DCollection(surf_trajectory.mesh.triangles)


    # https://www.jhuapl.edu/Content/techdigest/pdf/V23-N01/23-01-Holdridge.pdf -- NEAR Orbits
    for i in range(0, len(orbits)-1):
        poly_vis.plot_polyhedron(surf_trajectory.mesh, np.linalg.norm(a,axis=1))
        utc = [orbits[i], orbits[i+1]]
        trajectory = EphemerisDist("OREX", "BENNu", "BENNU_FIXED", orbits[i], orbits[i+1], 10*60)
        positions = trajectory.positions
        ax = plt.gca()
        ax.plot3D(positions[:,0], positions[:,1], positions[:,2], 'gray')
        
        ax.axes.set_xlim3d(left=np.min(positions[:,0]), right=np.max(positions[:,0])) 
        ax.axes.set_ylim3d(bottom=np.min(positions[:,1]), top=np.max(positions[:,1])) 
        ax.axes.set_zlim3d(bottom=np.min([np.min(positions[:,2]), -Eros().radius]), top=np.max([np.max(positions[:,2]), Eros().radius])) 
        poly_vis.save(plt.gcf(), directory + "OREX_Orbit_"+str(i)+ ".pdf")
    #plt.show()

if __name__ == '__main__':
    main()