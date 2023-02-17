import pandas as pd
from GravNN.Networks.Losses import *
import matplotlib.pyplot as plt
from DataInterface import DataInterface
from GravNN.Support.transformations import cart2sph
class AccelerationHistogram(DataInterface):
    def __init__(self, model, config, points, radius_bounds=None, random_seed=1234, remove_J2=False):
        super().__init__(model, config, points, radius_bounds=radius_bounds, random_seed=random_seed, remove_J2=remove_J2)
    
    def run(self):
        self.gather_data()
        self.plot()

    def plot(self):
        plt.figure()
        da = self.accelerations - self.predicted_accelerations
        plt.subplot(3,1,1)
        plt.hist(da[:,0], 250)
        plt.subplot(3,1,2)
        plt.hist(da[:,1], 250)
        plt.subplot(3,1,3)
        plt.hist(da[:,2], 250)
        plt.xlabel("Acceleration")

        from GravNN.Support.transformations import project_acceleration
        da_hill = project_acceleration(cart2sph(self.positions), da)
        plt.figure()
        plt.subplot(3,1,1)
        plt.hist(da_hill[:,0], 250)
        plt.subplot(3,1,2)
        plt.hist(da_hill[:,1], 250)
        plt.subplot(3,1,3)
        plt.hist(da_hill[:,2], 250)
        plt.xlabel("Sphere Acceleration")

        plt.figure()
        plt.hist(self.potentials - self.predicted_potentials, 1000)
        plt.xlabel("Potentials")
        plt.show()



def main():
    from GravNN.Networks.Model import load_config_and_model
    df_file = "Data/Dataframes/example.data"
    df = pd.read_pickle(df_file)
    model_id = df.iloc[-1]["id"]
    config, model = load_config_and_model(model_id, df)
    exp = AccelerationHistogram(model, config, 50000, remove_J2=True)
    exp.run()

    # plt.show()
if __name__ == "__main__":
    main()