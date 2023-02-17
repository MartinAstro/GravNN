import pandas as pd
from GravNN.Networks.Losses import *
import matplotlib.pyplot as plt
from DataInterface import DataInterface
from GravNN.Support.transformations import project_acceleration, cart2sph

class AltitudeNetworkOutput(DataInterface):
    def __init__(self, model, config, points, radius_bounds=None, random_seed=1234, remove_J2=False):
        super().__init__(model, config, points, radius_bounds=radius_bounds, random_seed=random_seed, remove_J2=remove_J2)
    
    def run(self):
        self.gather_data()
        self.plot()

    def plot(self):
        radius = np.linalg.norm(self.positions,axis=1)
        r_sph = cart2sph(self.positions)
        true_acc_sph = project_acceleration(r_sph, self.accelerations)
        pred_acc_sph = project_acceleration(r_sph, self.predicted_accelerations)

        plt.figure()
        plt.scatter(radius, true_acc_sph[:,0], label="true")
        plt.scatter(radius, pred_acc_sph[:,0], label="pred")
        plt.xlabel("Acceleration")

        plt.figure()
        plt.scatter(radius, self.potentials, label="true")
        plt.scatter(radius, self.predicted_potentials, label="pred")
        plt.xlabel("Potentials")

def main():
    from GravNN.Networks.Model import load_config_and_model
    df_file = "Data/Dataframes/example.data"
    df = pd.read_pickle(df_file)
    model_id = df.iloc[-1]["id"]
    config, model = load_config_and_model(model_id, df)
    exp = AltitudeNetworkOutput(model, config, 5000, remove_J2=False)
    exp.run()
    plt.show()
if __name__ == "__main__":
    main()