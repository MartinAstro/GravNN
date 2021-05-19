import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase

def main():
    vis = VisualizationBase()
    sh_df = pd.read_pickle('Data/Dataframes/sh_regress_stats.data')
    nn_df = pd.read_pickle('Data/Dataframes/nn_regress_stats.data')
    pinn_df = pd.read_pickle('Data/Dataframes/pinn_regress_stats.data')

    df_list = [sh_df, nn_df, pinn_df]
    marker = [None, None, 'o']
    linestyle = [None, '--', None]
    
    for noise in np.unique(sh_df.index.get_level_values(0)):
        vis.newFig()
        for i in range(len(df_list)):
            df = df_list[i]
            stats = df.loc[(noise)]
            x = stats['params']
            rse = stats['rse_mean'].values
            sigma = stats['sigma_2_mean'].values
            compliment = stats['sigma_2_c_mean'].values

            ax = sns.lineplot(x=x, y=rse, color='blue', linestyle=linestyle[i], marker=marker[i])
            ax = sns.lineplot(x=x, y=sigma, color='green', linestyle=linestyle[i], marker=marker[i])
            ax = sns.lineplot(x=x, y=compliment, color='red', linestyle=linestyle[i], marker=marker[i])


        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('RSE')
        plt.xlabel('Params')
        plt.legend(['A', 'F', 'C'])
        vis.save(plt.gcf(), 'Regression_'+str(noise)+".pdf")

    plt.show()
if __name__ == "__main__":
    main()