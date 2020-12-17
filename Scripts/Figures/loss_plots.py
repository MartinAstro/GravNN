from GravNN.Visualization.VisualizationBase import VisualizationBase
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import os
import sigfig

def loss_plots():
    with open('tensorflow_2_results.data', 'rb') as f:
        nn_df = pickle.load(f)
    histories = pd.Series()
    histories = histories.append(nn_df['history'].iloc[0:6])
    histories = histories.append(nn_df['history'].iloc[-6:-5])
    histories = histories.append(nn_df['history'].iloc[-3:-2])


    labels = ['Traditional', 'Traditional-PINN', 'Residual', 'Residual-PINN', 'Inception', 'Inception-PINN', 'Dense', 'Dense-PINN']
    colors = ['r', 'r', 'b', 'b', 'g', 'g', 'y', 'y']
    markers = ['D', '', 'D', '', 'D', '', 'D', '']

    vis = VisualizationBase("Plots/")
    fig, ax = vis.newFig()
    epochs = np.linspace(0, len(histories[0]['loss'][1000::500]), len(histories[0]['loss'][1000::500]))*500 + 1000
    for i in range(len(histories)):
        plt.plot(epochs, histories[i]['loss'][1000::500], label=labels[i], c=colors[i], marker=markers[i], markersize=3, markevery=50)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    vis.save(fig, "loss_plot.pdf")
    plt.show()

def results_tables():
    with open('tensorflow_2_results.data', 'rb') as f:
        nn_df = pickle.load(f)

    for i in range(len(nn_df)):
        nn_df['network_type'][i] = nn_df['network_type'][i].__name__

    columns = ['network_type', 'PINN_flag', 'sparsity', 'num_w_clusters','params', 'size', 'rse_median']

    print(nn_df.to_latex(columns=columns, index=False))

def ablation_table():
    with open('tensorflow_2_ablation.data', 'rb') as f:
        nn_df = pickle.load(f)

    for i in range(len(nn_df)):
        nn_df['network_type'][i] = nn_df['network_type'][i].__name__

    columns = ['network_type', 'epochs', 'batch_size', 'N_train', 'dropout', 'params', 'rse_median', 'sh_sigma_2_median', 'sh_sigma_2_c_median']

    print(nn_df.to_latex(columns=columns, index=False))
    

def main():
    #loss_plots()
    #results_tables()
    ablation_table()

if __name__=='__main__':
    main()