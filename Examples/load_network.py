import matplotlib.pyplot as plt
import pandas as pd
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.utils import print_config
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Networks.utils import get_history


def main():
    # Read the dataframe which contains the configuration data about the networks trained
    df = pd.read_pickle("Data/Dataframes/example_training.data")

    # Grab one of the model id's to be used to reinitialize the network
    model_id = df["id"].values[-1]

    # reload the configuration info and model
    config, model = load_config_and_model(model_id, df)

    # print an organized summary of hyperparameters and configuration details to the console
    print_config(config)

    # get the history from the network training and plot
    history = get_history(config['id'][0])

    # some formatting niceties
    vis = VisualizationBase()
    vis.newFig()
    plt.semilogy(history['loss'], color='blue', label='Loss')
    plt.semilogy(history['val_loss'], color='orange', label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()