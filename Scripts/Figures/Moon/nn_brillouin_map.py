        
import os

import matplotlib.pyplot as plt
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.Trajectories import DHGridDist, ReducedGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapBase import MapBase
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Networks.Model import load_config_and_model


def main():
    # Plotting 
    directory = os.path.abspath('.') +"/Plots/Moon/"
    os.makedirs(directory, exist_ok=True)

    map_vis = MapBase('mGal')
    map_vis.fig_size = map_vis.full_page
    #map_vis.tick_interval = [60, 60]

    my_cmap = 'viridis'
    vlim= [0, 60]

    planet = Moon()
    model_file = planet.sh_file
    density_deg = 180

    df_file ='C:\\Users\\John\\Documents\\Research\\ML_Gravity\\Data\\Dataframes\\moon_pinn_df.data'
    df = pd.read_pickle(df_file)

    trajectory = DHGridDist(planet, planet.radius, degree=density_deg)
    
    for i in range(len(df)):
        row = df.iloc[i]
        model_id = row['id']
        config, model = load_config_and_model(model_id, df)

        x_transformer = config['x_transformer'][0]
        a_transformer = config['a_transformer'][0]

        x = x_transformer.transform(trajectory.positions)
        output = model.predict(x)
        a = model.get_acceleration(x)

        a_pred = a_transformer.inverse_transform(a)
        grid_true = Grid(trajectory=trajectory, accelerations=a_pred)
        map_vis.plot_grid(grid_true.total, vlim=vlim, label='[mGal]', extend='max')#"U_{1000}^{(2)} - U_{100}^{(2)}")
        map_vis.save(plt.gcf(), directory + "pinn_brillouin_" + str(row['num_units']) + ".pdf")


    plt.show()
if __name__ == "__main__":
    main()
