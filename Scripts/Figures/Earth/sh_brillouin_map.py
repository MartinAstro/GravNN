        
import os

import matplotlib.pyplot as plt
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Support.Grid import Grid
from GravNN.Trajectories import DHGridDist
from GravNN.Visualization.MapBase import MapBase


def main():
    
    planet = Earth()
    model_file = planet.sh_file
    density_deg = 180
    max_deg = 1000

    radius_min = planet.radius
    
    df_file = "Data/Dataframes/sh_stats_Brillouin.data"
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    Call_a = Call_r0_gm.load().accelerations

    C55_r0_gm = SphericalHarmonics(model_file, degree=55, trajectory=trajectory)
    C55_a = C55_r0_gm.load().accelerations

    C110_r0_gm = SphericalHarmonics(model_file, degree=110, trajectory=trajectory)
    C110_a = C110_r0_gm.load().accelerations

    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
    C22_a = C22_r0_gm.load().accelerations
        

    


    directory = os.path.abspath('.') +"/Plots/OneOff/"
    os.makedirs(directory, exist_ok=True)

    mapUnit = 'mGal'
    map_vis = MapBase(mapUnit)
    map_vis.fig_size = map_vis.full_page

    vlim= [0, 30]
    grid_true = Grid(trajectory=trajectory, accelerations=Call_a-C22_a)
    map_vis.plot_grid(grid_true.total, vlim=vlim, label=None)
    map_vis.save(plt.gcf(), directory + "sh_brillouin_true_map.pdf")


    vlim= [0, 30]
    grid_true = Grid(trajectory=trajectory, accelerations=C55_a-C22_a)
    map_vis.plot_grid(grid_true.total, vlim=vlim, label=None)
    map_vis.save(plt.gcf(), directory + "sh_brillouin_55_map.pdf")


    # For moon comparision
    map_vis.fig_size = map_vis.half_page
    map_vis.tick_interval = [60, 60]
    map_vis.newFig()
    vlim= [0, 40]
    grid_true = Grid(trajectory=trajectory, accelerations=Call_a-C22_a)
    im = map_vis.new_map(grid_true.total, vlim=vlim, cmap='viridis')#,log_scale=True)
    map_vis.add_colorbar(im, '[mGal]', vlim=vlim, extend='max', loc='top', orientation='horizontal', pad=0.05)    
    map_vis.save(plt.gcf(), directory + "sh_brillouin_true_map_half.pdf")


    # LEO Altitude 
    trajectory = DHGridDist(planet, radius_min+420000.0, degree=density_deg)

    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    Call_a = Call_r0_gm.load().accelerations

    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
    C22_a = C22_r0_gm.load().accelerations
        
    vlim= [0, 5]
    map_vis.fig_size = map_vis.full_page
    grid_true = Grid(trajectory=trajectory, accelerations=Call_a-C22_a)
    map_vis.plot_grid(grid_true.total, vlim=vlim, label='[mGal]')
    map_vis.save(plt.gcf(), directory + "sh_brillouin_LEO_true_map.pdf")


    plt.show()
if __name__ == "__main__":
    main()
