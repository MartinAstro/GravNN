from GravNN.Networks.utils import _get_loss_fcn
from GravNN.Networks.Data import DataSet
from GravNN.Trajectories.RandomDist import RandomDist
import numpy as np
import pandas as pd
from GravNN.Networks.Losses import *
from pprint import pprint
from DataInterface import DataInterface

class MetricsExperiment(DataInterface):
    def __init__(self,
                 model,
                 config,
                 points,
                 radius_bounds=None,
                 random_seed=1234,
                 remove_J2=False):
        super().__init__(
            model,
            config,
            points,
            radius_bounds=radius_bounds,
            random_seed=random_seed,
            remove_J2=remove_J2)

    def compute_losses(self, loss_fcn_list):
        losses = {}
        for loss_key in loss_fcn_list:
            loss_fcn = get_loss_fcn(loss_key)
            
            # Compute loss on acceleration and potential
            losses.update({
                f"{loss_fcn.__name__}" : loss_fcn(
                    self.predicted_accelerations - self.LF_accelerations, 
                    self.accelerations - self.LF_accelerations
                    )
                })
        self.losses = losses
    
    def compute_metrics(self):
        metrics = {}
        for key, value in self.losses.items():
            metrics.update({
                f"{key}_mean" : np.mean(value),
                f"{key}_std" : np.std(value),
                f"{key}_max" : np.max(value),
            })
        self.metrics = metrics

    def run(self, loss_fcn_list):
        self.get_data()
        self.get_PINN_data()
        self.get_J2_data()
        self.compute_losses(loss_fcn_list)
        self.compute_metrics()

    def save_metrics_in_df(self, df, i):
        # populate dataframe keys s.t. they can be updated
        for metrics_key in self.metrics.keys():
            if metrics_key not in df.columns:
                df[metrics_key] = None

        idx = df.index[i]
        new_col = pd.DataFrame(self.metrics, index=[idx])
        df.update(new_col)
        return df


def main():
    from GravNN.Networks.Model import load_config_and_model
    df_file = "Data/Dataframes/fourier_features_search.data"
    df_file_new = df_file.split(".data")[0] + "_metrics.data"
    df = pd.read_pickle(df_file)

    for i in range(len(df)):
        model_id = df.iloc[i]["id"]
        config, model = load_config_and_model(model_id, df)
        config['override'] = [False]
        metrics_exp = MetricsExperiment(model, config, 50000, remove_J2=True)
        loss_fcn_list = ['rms', 'percent', 'angle', 'magnitude']
        metrics_exp.run(loss_fcn_list)
        pprint(metrics_exp.metrics)
        df = metrics_exp.save_metrics_in_df(df, i)

    df.to_pickle(df_file_new)

    # plt.show()
if __name__ == "__main__":
    main()