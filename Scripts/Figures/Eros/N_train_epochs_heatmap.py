from GravNN.Visualization.HeatmapVisualizer import Heatmap3DVisualizer
import pandas as pd
import matplotlib.pyplot as plt

def PINN_III():
    df_file = "Data/Dataframes/eros_PINN_III_hparams_metrics.data"
    df = pd.read_pickle(df_file)

    df.percent_mean = df.percent_mean.astype(float)
    v_min = df['percent_mean'].min()
    v_max = df.nlargest(2, 'percent_mean')['percent_mean'].values[1]#['percent_mean'].max()

    vis = Heatmap3DVisualizer(df)
    query = "num_units == 10"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "PINNIII/Eros_NvE_10.pdf")

    query = "num_units == 20"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "PINNIII/Eros_NvE_20.pdf")

    query = "num_units == 40"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "PINNIII/Eros_NvE_40.pdf")

    query = "num_units == 80"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "PINNIII/Eros_NvE_80.pdf")

    

def PINN_II():
    df_file = "Data/Dataframes/eros_PINN_II_hparams_metrics.data"
    df = pd.read_pickle(df_file)

    df_file_PINN_III = "Data/Dataframes/eros_PINN_III_hparams_metrics.data"
    df_PINN_III = pd.read_pickle(df_file_PINN_III)

    # scale by PINN III Results
    df_PINN_III.percent_mean = df_PINN_III.percent_mean.astype(float)
    v_min = df_PINN_III['percent_mean'].min()
    v_max = df_PINN_III.nlargest(2, 'percent_mean')['percent_mean'].values[1]#['percent_mean'].max()

    vis = Heatmap3DVisualizer(df)
    query = "num_units == 10"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "PINNII/Eros_NvE_10.pdf")

    query = "num_units == 20"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "PINNII/Eros_NvE_20.pdf")

    query = "num_units == 40"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "PINNII/Eros_NvE_40.pdf")

    query = "num_units == 80"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "PINNII/Eros_NvE_80.pdf")



def main():
    PINN_III()
    PINN_II()
    plt.show()

if __name__ == "__main__":
    main()