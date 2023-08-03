import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.Visualization.VisualizationBase import VisualizationBase


class WeightsVisualizer(VisualizationBase):
    def __init__(self, model, config, **kwargs):
        super().__init__(**kwargs)
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.model = model
        self.config = config

        self.weights, self.biases = self.gather_weights()

    def get_layer_number(self, name):
        words = name.split("_")
        if len(words) == 2:
            i = int(words[1][0])
        else:
            i = 0
        return i

    def gather_weights(self):
        params = self.model.network.trainable_variables
        weights = {}
        biases = {}
        for layer in params:
            idx = self.get_layer_number(layer.name)
            if "kernel" in layer.name:
                weights.update({str(idx): layer.numpy().flatten()})
            if "bias" in layer.name:
                biases.update({str(idx): layer.numpy().flatten()})
        return weights, biases

    def plot_hist(self, dict_params, label):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        nbins = 50

        for idx, values in dict_params.items():
            ys = values

            hist, bins = np.histogram(ys, bins=nbins)
            xs = (bins[:-1] + bins[1:]) / 2

            ax.bar(xs, hist, width=np.diff(bins)[0], zs=int(idx), zdir="y", alpha=0.5)

        ax.set_xlabel(label)
        ax.set_ylabel("Dense Layer Idx")
        ax.set_zlabel("Frequency")

    def plot(self):
        self.plot_hist(self.weights, "Weights")
        self.plot_hist(self.biases, "Biases")


5
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from GravNN.Networks.Model import load_config_and_model

    df = pd.read_pickle("Data/Dataframes/test.data")
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(df, model_id)

    vis = WeightsVisualizer(model, config)
    vis.plot()
    plt.show()
