from multiprocessing.sharedctypes import Value
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Support.transformations import sphere2cart, cart2sph
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
import pandas as pd
from GravNN.Networks.Data import DataSet

class TrainingDistributionVisualizer(VisualizationBase):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.data = DataSet(config)
        self.planet_radius = config['planet'][0].radius
        self.radius_min = config['radius_min'][0]
        self.radius_max = config['radius_max'][0]


    def plot_3d_scatter(self):
        self.new3DFig()
        X = self.data.raw_data['x_train'] / self.planet_radius
        x = X[:,0]
        y = X[:,1]
        z = X[:,2]
        plt.gca().scatter(x, y, z, alpha=0.2, s=2)
        plt.gca().set_xlabel('x [R]')
        plt.gca().set_ylabel('y [R]')
        plt.gca().set_zlabel('z [R]')

    def plot_histogram(self):
        self.newFig()
        r = np.linalg.norm(self.data.raw_data['x_train'], axis=1) / self.planet_radius
        plt.hist(r, 50, alpha=0.2)
        plt.ylabel('Frequency')
        plt.xlabel('Radius [R]')

    def plot(self):
        self.plot_histogram()
        self.plot_3d_scatter()



if __name__ == "__main__":
    df_file = "Data/Dataframes/test.data" 
    import pandas as pd
    import numpy as np
    from GravNN.Visualization.TrainingDistributionVisualizer import TrainingDistributionVisualizer
    import matplotlib.pyplot as plt
    from GravNN.Networks.Model import load_config_and_model

    df = pd.read_pickle("Data/Dataframes/test.data")
    model_id = df["id"].values[-1] 
    config, model = load_config_and_model(model_id, df)
    vis = TrainingDistributionVisualizer(config)
    vis.plot()
    plt.show()
