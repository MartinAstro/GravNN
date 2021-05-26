import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase

def plot_model(vis, stat):
    sh_df = pd.read_pickle('Data/Dataframes/sh_moon_regress_stats.data')
    nn_df = pd.read_pickle('Data/Dataframes/nn_moon_regress_stats.data')
    pinn_df = pd.read_pickle('Data/Dataframes/pinn_moon_regress_stats.data')

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
            info = stats[stat].values
            ax = sns.lineplot(x=x, y=info, color=color[noise], linestyle=linestyle[i], marker=marker[i], ci=99)


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
    vis.save(plt.gcf(), 'Regression_moon.pdf')
    plot_model(vis, 'sigma_2_mean')
    vis.save(plt.gcf(), 'Regression_sigma_moon.pdf')

    plt.show()
if __name__ == "__main__":
    main()