import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(df, x, y, z):
    data = df.pivot(index=x, columns=y, values=z).astype(float)
    sns.heatmap(data, cmap=sns.color_palette("coolwarm", as_cmap=True))
    plt.title(z)

def main():
    df = pd.read_pickle("Data/Dataframes/earth_all2.data")

    plt.figure()
    plt.subplot(221)
    plot_heatmap(df, 'N_train', 'num_units', 'loss')
    plt.subplot(222)
    plot_heatmap(df, 'N_train', 'num_units', 'percent_error')
    plt.subplot(223)
    plot_heatmap(df, 'N_train', 'num_units', 'RMS')
    plt.subplot(224)
    plot_heatmap(df, 'N_train', 'num_units', 'w_percent_error')
    plt.show()

if __name__ == "__main__":
    main()