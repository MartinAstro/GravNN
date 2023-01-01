from GravNN.Visualization.HeatmapVisualizer import Heatmap3DVisualizer
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df_file = "Data/Dataframes/epochs_N_search_all_metrics.data"
    df = pd.read_pickle(df_file)

    v_min = df['percent_mean'].min()
    v_max = df['percent_mean'].max()

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
    vis.save(plt.gcf(), "Plots/PINNIII/NvE_10.pdf")

    query = "num_units == 20"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "Plots/PINNIII/NvE_20.pdf")

    query = "num_units == 40"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "Plots/PINNIII/NvE_40.pdf")

    query = "num_units == 80"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
    vis.save(plt.gcf(), "Plots/PINNIII/NvE_80.pdf")

    plt.show()


if __name__ == "__main__":
    main()