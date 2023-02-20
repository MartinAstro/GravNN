from GravNN.Visualization.VisualizationBase import VisualizationBase
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.utils import get_history


class HistoryVisualizer(VisualizationBase):
    def __init__(self, model, config, **kwargs):
        super().__init__(**kwargs)
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.model = model
        self.config = config

        self.get_history()

    def get_history(self):
        try:
            self.history = get_history(self.config['id'][0])
        except:
            self.history = self.config['history'][0]
        self.loss = self.history['loss']
        self.val_loss = self.history['val_loss']
        self.percent_error = self.history['percent_mean']
        self.val_percent_error = self.history['val_percent_mean']
        self.epochs = np.arange(0, len(self.loss), 1)

    def plot(self, x, y, log_x=False, log_y=False, **kwargs):
        if log_x and log_y:
            plt_fcn = plt.loglog
        elif log_x:
            plt_fcn = plt.semilogx
        elif log_y:
            plt_fcn = plt.semilogy
        else:
            plt_fcn = plt.plot
        label = kwargs.get('label', "loss")
        linestyle= kwargs.get('linestyle', '-')
        color = kwargs.get('color', None)
        alpha = kwargs.get('alpha', 1.0)
        line = plt_fcn(x, y, 
                label=label, 
                linestyle=linestyle, 
                color=color,
                alpha=alpha)
        plt.grid()
        plt.tight_layout()
        return line

    def plot_loss(self, **kwargs):
        if kwargs.get('new_fig', True):
            plt.figure()
        # Plot loss
        line = self.plot(self.epochs, self.loss, **kwargs)
        
        # update validation keywords
        kwargs.update({"label" : "val " + kwargs.get('label', "loss")})

        # plot validation
        self.plot(self.epochs, self.val_loss, 
                    linestyle='--',
                    alpha=0.5, 
                    color=line[0].get_color(),
                    **kwargs)
        skip_epochs = kwargs.get("skip_epochs", 20)
        plt.ylim([None, self.val_loss[skip_epochs]])
        plt.legend()

<<<<<<< HEAD
if __name__ == "__main__":
    df, idx = pd.read_pickle('Data/Dataframes/big_net_run.data'), -1 #i = 105 is best
=======
def main():
    df, idx = pd.read_pickle('Data/Dataframes/example.data'), -1 #i = 105 is best
>>>>>>> 8672f90 (Add main to scripts to call after training loop)

    model_id = df["id"].values[idx]
    config, model = load_config_and_model(model_id, df)

    vis = HistoryVisualizer(model, config)
    vis.plot_loss(log_y=False)
    plt.show()

if __name__ == "__main__":
    main()