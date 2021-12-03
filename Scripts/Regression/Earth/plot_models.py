import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase

def plot_model(vis, stat):
    vis = VisualizationBase()
    sh_df = pd.read_pickle('Data/Dataframes/sh_regress_2_stats.data')
    nn_df = pd.read_pickle('Data/Dataframes/nn_regress_4_stats.data')
    pinn_df = pd.read_pickle('Data/Dataframes/pinn_regress_4_stats.data')

    # nn_df = pd.read_pickle('Data/Dataframes/nn_regress_stats.data')
    # pinn_df = pd.read_pickle('Data/Dataframes/pinn_regress_stats.data')



    # the new spherical harmonic models use noise as 0.0 and 0.2, whereas the networks use 0 and 2
    nn_index = [(entry[0], entry[1],entry[2]) for entry in nn_df.index.values]
    pinn_index = [(entry[0], entry[1],entry[2]) for entry in pinn_df.index.values]
    nn_df.set_index([nn_index], inplace=True)
    pinn_df.set_index([pinn_index], inplace=True)
 
    df_list = [sh_df, nn_df, pinn_df]
    marker = [None, None, 'o']
    linestyle = [None, '--', None]
    color = ['blue', 0, 'red']
    vis.newFig()
    lines = []
    for noise in np.unique(sh_df.index.get_level_values(0)):
        lines_styles = []
        for i in range(len(df_list)):
            df = df_list[i]
            try:
                stats = df.loc[(noise)].drop(40)
            except:
                stats = df.loc[(noise)]
            x = stats['params']
            rse = stats['rse_mean'].values
            sigma = stats['sigma_2_mean'].values
            compliment = stats['sigma_2_c_mean'].values
            ax = sns.lineplot(x=x, y=rse, color=color[int(noise*10)], linestyle=linestyle[i], marker=marker[i], ci=99)
            #ax = sns.lineplot(x=x, y=sigma, color=color[noise], linestyle=linestyle[i], marker=marker[i])
            #ax = sns.lineplot(x=x, y=compliment, color=color[noise], linestyle=linestyle[i], marker=marker[i])
            lines_styles.append(ax.lines[-1])
        lines.append(lines_styles)

    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylabel('MRSE [m/s$^2$]')
    plt.xlabel('Parameters')
    legend1 = plt.legend(lines[0], ['SH', 'NN', 'PINN'], loc=1)
    plt.legend([l[0] for l in lines], ['$0\sigma$', '$2\sigma$'], loc=3)
    plt.gca().add_artist(legend1)


def main():

    vis = VisualizationBase()
    plot_model(vis, 'rse_mean')
    vis.save(plt.gcf(), 'Regression.pdf')
    plot_model(vis, 'sigma_2_mean')
    vis.save(plt.gcf(), 'Regression_sigma.pdf')

    plt.show()
if __name__ == "__main__":
    main()