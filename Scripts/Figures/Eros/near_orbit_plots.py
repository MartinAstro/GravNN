
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
    orbits = ['Feb 24, 2000', 
            'Mar 03, 2000',
            'Apr 02, 2000',
            'Apr 11, 2000',
            'Apr 22, 2000',
            'Apr 30, 2000',
            'July 07, 2000',
            'July 14, 2000',
            'July 24, 2000',
            'July 31, 2000',
            'Aug 08, 2000',
            'Aug 26, 2000',
            'Sep 05, 2000',
            'Oct 13, 2000',
            'Oct 20, 2000',
            'Oct 25, 2000',
            'Oct 26, 2000',
            'Nov 03, 2000',
            'Dec 07, 2000',
            'Dec 13, 2000',
            'Jan 24, 2001',
            'Jan 28, 2001',
            'Jan 29, 2001',
            'Feb 02, 2001',
            'Feb 06, 2001']

    # get et values one and two, we could vectorize str2et
    vis = VisualizationBase()
    poly_vis = PolyVisualization()
    surf_trajectory = SurfaceDist(Eros(), Eros().obj_8k)
    x, a, u = get_poly_data(surf_trajectory, Eros().obj_8k)
    tri = Poly3DCollection(surf_trajectory.mesh.triangles)


    # https://www.jhuapl.edu/Content/techdigest/pdf/V23-N01/23-01-Holdridge.pdf -- NEAR Orbits
    for i in range(0, len(orbits)-1):
        poly_vis.plot_polyhedron(surf_trajectory.mesh, np.linalg.norm(a,axis=1),cbar=False)
        utc = [orbits[i], orbits[i+1]]
        trajectory = EphemerisDist("NEAR", "EROS", "EROS_FIXED", orbits[i], orbits[i+1], 10*60)
        positions = trajectory.positions
        ax = plt.gca()
        ax.plot3D(positions[:,0], positions[:,1], positions[:,2], 'gray')
        
        # ax.axes.set_xlim3d(left=np.min(positions[:,0]), right=np.max(positions[:,0])) 
        # ax.axes.set_ylim3d(bottom=np.min(positions[:,1]), top=np.max(positions[:,1])) 
        # ax.axes.set_zlim3d(bottom=np.min([np.min(positions[:,2]), -Eros().radius]), top=np.max([np.max(positions[:,2]), Eros().radius])) 

        max = np.max(positions)
        min = np.min(positions)

        bound = np.max(np.abs([max, min]))

        ax.axes.set_xlim3d(left=-bound, right=bound) 
        ax.axes.set_ylim3d(bottom=-bound, top=bound) 
        ax.axes.set_zlim3d(bottom=-bound, top=bound) 
        if i == 1 or i == 7 or i == 9 or i == 19:
            poly_vis.save(plt.gcf(), directory + "NEAR_Orbit_"+str(i)+ ".pdf")
    #plt.show()

if __name__ == '__main__':
    main()