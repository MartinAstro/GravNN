import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase

def plot_model(sh_df, nn_df, pinn_df, metric, plot_name):
    vis = VisualizationBase()

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
            # if noise == 0.2:
            #     continue

            df = df_list[i]
            try:
                stats = df.loc[(noise)].drop(40)
            except:
                stats = df.loc[(noise)]
            x = stats['params']
            
            metric_values = stats[metric].values
            ax = sns.lineplot(x=x, y=metric_values, color=color[int(noise*10)], linestyle=linestyle[i], marker=marker[i], ci=95)
            lines_styles.append(ax.lines[-1])
        lines.append(lines_styles)

    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylabel('MRSE [m/s$^2$]')
    plt.xlabel('Parameters')
    legend1 = plt.legend(lines[0], ['SH', 'NN', 'PINN'], loc=1)
    #plt.legend([l[0] for l in lines], ['$0\sigma$', '$2\sigma$'], loc=2)
    plt.gca().add_artist(legend1)
    vis.save(plt.gcf(), plot_name)


def main():

    vis = VisualizationBase()

    # 9,500 -- with proper preprocessing & better hyperparams (gelu, higher learning rate, larger validation set)

    # sh_df = pd.read_pickle('Data/Dataframes/Regression/Earth_SH_regression_9500_v1_stats.data') # only up to a degree 40 model with 3 trials
    sh_df = pd.read_pickle('Data/Dataframes/Regression/Moon_SH_regression_5000_v1_stats.data')
    #     nn_file = 'Data/Dataframes/Regression/Moon_NN_regression_5000_v1'
    # pinn_file = 'Data/Dataframes/Regression/Moon_PINN_regression_5000_v1'
    nn_df = pd.read_pickle('Data/Dataframes/Regression/Moon_NN_regression_5000_v2_stats.data')
    pinn_df = pd.read_pickle('Data/Dataframes/Regression/Moon_PINN_regression_5000_v2_stats.data')

    # sh_df = pd.read_pickle('Data/Dataframes/Regression/Earth_SH_regression_5000_v1_stats.data') #gelu w/ exp decay
    # nn_df = pd.read_pickle('Data/Dataframes/Regression/Earth_NN_regression_9500_v4_stats.data')
    # pinn_df = pd.read_pickle('Data/Dataframes/Regression/Earth_PINN_regression_9500_v4_stats.data')

    plot_model(sh_df, nn_df, pinn_df, 'rse_mean', 'Moon_Regression.pdf')
    plot_model(sh_df, nn_df, pinn_df, 'sigma_2_mean', 'Moon_Regression_sigma.pdf')

    plt.show()
if __name__ == "__main__":
    main()