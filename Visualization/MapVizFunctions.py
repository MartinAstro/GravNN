from Visualization.MapVisualization import MapVisualization
import numpy as np
import matplotlib.pyplot as plt
def high_fidelity_maps(true_grid, C20_grid):
    """
    Generates highest fidelity map of the planet gravity field provided. This includes the full map as well as the perturbations above C20.
    """
    map_viz = MapVisualization()
    true_mC20_grid = true_grid - C20_grid
    map_viz.plot_grid(true_grid.total, "Acceleration Total")
    map_viz.plot_grid(true_mC20_grid.total, "Acceleration Perturbations")
    return

def percent_error_maps(true_grid, sub_grid, C20_grid, vlim=None):
    """
    Shows the relative error maps of gravity field (relative to ground truth and sub C20 signals)
    """
    map_viz = MapVisualization()
    Call_mC20_grid = true_grid - C20_grid
    delta_sub_grid = sub_grid - C20_grid

    # Absolute Error (low degree model error w.r.t. full gravity model)
    map_viz.plot_grid_error(sub_grid.total, true_grid.total, vlim=None)#100)

    # Relative Error (low degree - C20 w.r.t full - C20)
    map_viz.plot_grid_error(delta_sub_grid.total, Call_mC20_grid.total, vlim=vlim)
    return

def component_error(degree_list, radius, file_name, true_grid, C20_grid):
    """
    Shows the relative error of acceleration components 
    """
    error_list = np.zeros((len(degree_list),4))
    map_grid = DHGridDist(planet, radius, degree=175)

    for k in range(len(degree_list)):
        Clm_gm = SphericalHarmonics(file_name, degree_list[k], trajectory=map_grid)
        sub_grid = Grid(gravityModel=Clm_gm)
        Call_mC20_grid = true_grid - C20_grid
        delta_sub_grid = sub_grid - C20_grid

        error_grid = (delta_sub_grid - Call_mC20_grid)/Call_mC20_grid * 100
        error_list[k,0] = np.average(abs(error_grid.total))
        error_list[k,1] = np.average(abs(error_grid.r))
        error_list[k,2] = np.average(abs(error_grid.theta))
        error_list[k,3] = np.average(abs(error_grid.phi))

    map_viz = MapVisualization()
    map_viz.plot_component_errors(degree_list, error_list, 'blue')
    return