import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    vis = VisualizationBase()
    vis.fig_size = (vis.w_full, vis.h_full / 4)

    x_min, x_max = 0, 7
    x_boundary = 5
    x_analytic = 2
    x = np.linspace(x_min, x_max, 100)

    def H(x, k, r):
        return (1 + np.tanh(k * (x - r))) / 2

    def G(x, k, r):
        return 1 - H(x, k, r)

    k = 2
    r = x_boundary

    r_ana = H(x, k, x_analytic)
    r_int = H(x, k, r)
    r_ext = G(x, k, r)

    vis.newFig()
    plt.plot(
        x,
        r_ana,
        color="green",
        linewidth=1.25,
        label=r"$w_{\text{LF}}$",
    )
    plt.plot(
        x,
        r_int,
        color="blue",
        linewidth=1.25,
        label=r"$w_{\text{BC}}$",
    )
    plt.plot(
        x,
        r_ext,
        color="red",
        linewidth=1.25,
        label=r"$w_{\text{NN}}$",
    )

    mask = x > x_boundary
    mask_inv = np.all(np.block([[x < x_boundary], [x > 1]]), axis=0)
    mask_int = x < 1

    # Shade the region where r < 1
    y_max = 1
    y_min = 0
    plt.fill_between(
        x[mask_int],
        y_min,
        y_max,
        color="yellow",
        alpha=0.2,
    )
    plt.fill_between(
        x[mask_inv],
        y_min,
        y_max,
        color="green",
        alpha=0.2,
    )
    plt.fill_between(
        x[mask],
        y_min,
        y_max,
        color="red",
        alpha=0.2,
    )

    # add a legend for the colors in upper left
    legend_2 = plt.legend(
        handles=[
            Rectangle((0, 0), 1, 1, alpha=0.2, color="yellow", label="Interior"),
            Rectangle((0, 0), 1, 1, alpha=0.2, color="green", label="Exterior"),
            Rectangle((0, 0), 1, 1, alpha=0.2, color="red", label="Extrapolation"),
        ],
        loc="upper left",
    )
    plt.gca().add_artist(legend_2)

    # make a vertical line at x = 2
    eps = 0.1
    plt.axvline(x=1.9, color="black", linestyle="--", linewidth=0.5)
    plt.text(
        1.9 + eps,
        0.1,
        r"$r^{\star} = 1 + e$",
        rotation=90,
        color="black",
        fontsize=6,
        bbox=dict(boxstyle="round", fc="w", alpha=0.7),
    )

    plt.axvline(x=x_boundary, color="black", linestyle="--", linewidth=0.5)
    plt.text(
        x_boundary + eps,
        0.1,
        "Training Data Limit",
        rotation=90,
        color="black",
        fontsize=6,
        bbox=dict(boxstyle="round", fc="w", alpha=0.7),
    )

    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])
    plt.xlabel("Radius [R]")
    plt.ylabel("Model Weight [-]")

    plt.legend()

    vis.save(plt.gcf(), "BoundaryCondition_v2")

    plt.show()


if __name__ == "__main__":
    main()
