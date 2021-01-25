import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/")
    vis.fig_size = vis.full_page
    df = pd.read_pickle('Data/speed_results_v2.data')

    poly_time = df['poly_time']
    sh_time = df['sh_time']
    nn_time = df['nn_time']
    pinn_time = df['pinn_time']


    fig, ax = vis.newFig()
    plt.loglog(df.index, poly_time, label='Poly')
    plt.loglog(df.index, sh_time, label='SH')
    plt.loglog(df.index, nn_time, label='NN (GPU)')
    plt.loglog(df.index, pinn_time, label='PINN (GPU)')

    nn_CPU_time = df['nn_cpu_time']
    pinn_CPU_time = df['pinn_cpu_time']
    plt.loglog(df.index, nn_CPU_time, label='NN (CPU)')
    plt.loglog(df.index, pinn_CPU_time, label='PINN (CPU)')
    plt.legend()

    plt.xlabel('Parameters')
    plt.ylabel('Time [s]')

    vis.save(fig, 'OneOff/speed_plot.pdf')
    #df.plot(logy=True, logx=True)
    plt.show()
if __name__ == '__main__':
    main()