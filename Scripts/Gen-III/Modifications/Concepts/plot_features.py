import matplotlib.pyplot as plt
import numpy as np

from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    vis = VisualizationBase()

    r = np.linspace(0, 3, 100)
    r_inv = 1 / r

    r_int = np.clip(r, 0, 1)
    r_ext = np.clip(r_inv, 0, 1)

    vis.newFig()
    plt.plot(r, r_int, color="blue", label="$r_i$", linewidth=1.5)
    plt.plot(r, r_ext, color="red", label="$r_e$", linewidth=1.5)

    mask = r > 1
    mask_inv = r < 1

    plt.plot(r[mask], r[mask], color="black", linestyle="--")
    plt.plot(r[mask_inv], r_inv[mask_inv], color="black", linestyle="--")

    # Shade the region where r < 1
    y_max = 2
    y_min = 0
    plt.fill_between(
        r[mask_inv],
        y_min,
        y_max,
        color="yellow",
        alpha=0.2,
    )  # , label='Interior')
    plt.fill_between(
        r[mask],
        y_min,
        y_max,
        color="green",
        alpha=0.2,
    )  # , label='Exterior')

    # label the interior and exterior regions
    plt.text(
        1.75,
        1.5,
        "Exterior",
        color="black",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="w"),
    )
    plt.text(
        0.5,
        0.25,
        "Interior",
        color="black",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="w"),
    )

    plt.ylim([y_min, y_max])
    plt.xlim([0, 3])
    plt.xlabel("Radius [R]")
    plt.ylabel("Feature Value")

    plt.legend()

    vis.save(plt.gcf(), "features")

    plt.show()


if __name__ == "__main__":
    main()
