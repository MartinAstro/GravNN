import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/")
    vis.fig_size = vis.full_page
    df = pd.read_pickle('Data/Dataframes/speed_results_journal.data')
    #df = pd.read_pickle('Data/speed_results_v2.data')

    fig, ax = vis.newFig()
    plt.loglog(df.index, df['poly_time'], label='Poly')
    plt.loglog(df.index, df['sh_time'], label='SH')
    plt.loglog(df.index, df['nn_time'], label='NN (GPU)')
    plt.loglog(df.index, df['pinn_time'], label='PINN (GPU)')

    try: 
        plt.loglog(df.index, df['nn_cpu_time'], label='NN (CPU)')
        plt.loglog(df.index, df['pinn_cpu_time'], label='PINN (CPU)')
    except:
        pass
    plt.legend()

    plt.xlabel('Parameters')
    plt.ylabel('Time [s]')

    vis.save(fig, 'OneOff/speed_plot.pdf')
    plt.show()
if __name__ == '__main__':
    main()