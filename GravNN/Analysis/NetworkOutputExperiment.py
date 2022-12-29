from GravNN.Networks.utils import _get_loss_fcn
from GravNN.Networks.Data import DataSet
from GravNN.Trajectories.RandomDist import RandomDist
import numpy as np
import pandas as pd
from GravNN.Networks.Losses import *
from pprint import pprint
from MetricsExperiment import MetricsExperiment
import matplotlib.pyplot as plt

class NetworkOutputExperiment:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.metric_exp = MetricsExperiment(model, config, 50000)

    def run(self):
        self.metric_exp.run(['rms', 'percent', 'angle', 'magnitude'])
        self.plot_potentials()
        self.plot_u_nn()


    def plot_potentials(self):
        plt.figure()
        x = self.metric_exp.positions
        u_hat = self.metric_exp.predicted_potentials
        u_true = self.metric_exp.potentials

        r = np.linalg.norm(x, axis=1)
        plt.scatter(r, u_hat, label='predicted')
        plt.scatter(r, u_true, label='true')
        plt.legend()


    def plot_u_nn(self):
        x = self.metric_exp.positions

        x_transformer = self.model.config["x_transformer"][0]
        u_transformer = self.model.config["u_transformer"][0]
        x = x_transformer.transform(x)
        # u_nn = self.model.network(x)
        outputs = self.model.network.layers[-2].output
        new_model = tf.keras.Model(inputs=self.model.network.input, outputs=outputs)
        u_nn = new_model(x)

        r = np.linalg.norm(x,axis=1)
        plt.figure()
        plt.scatter(r, u_nn)

    

def main():
    from GravNN.Networks.Model import load_config_and_model
    df_file = "Data/Dataframes/earth_high_alt4.data"
    df = pd.read_pickle(df_file)
    i = -1
    model_id = df.iloc[i]["id"]
    config, model = load_config_and_model(model_id, df)
    exp = NetworkOutputExperiment(model, config)
    exp.run()
    plt.show()

if __name__ == "__main__":
    main()