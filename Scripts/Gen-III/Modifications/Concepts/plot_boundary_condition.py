import matplotlib.pyplot as plt
import numpy as np

from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    vis = VisualizationBase()

    x_min, x_max = 3, 7
    x_boundary = 5
    x = np.linspace(x_min, x_max, 100)

    def H(x, k, r):
        return (1 + np.tanh(k * (x - r))) / 2

    def G(x, k, r):
        return 1 - H(x, k, r)

    k = 2
    r = x_boundary

    r_int = H(x, k, r)
    r_ext = G(x, k, r)

    vis.newFig()
    plt.plot(
        x,
        r_int,
        color="blue",
        linewidth=1.5,
        label=r"$w_{\text{LF}}$",
    )  # , label='$H(x)$')
    plt.plot(
        x,
        r_ext,
        color="red",
        linewidth=1.5,
        label=r"$w_{\text{NN}}$",
    )  # , label='$G(x)$')

    mask = x > x_boundary
    mask_inv = x < x_boundary

    # Shade the region where r < 1
    y_max = 1
    y_min = 0
    plt.fill_between(
        x[mask_inv],
        y_min,
        y_max,
        color="green",
        alpha=0.2,
    )  # , label='Interior')
    plt.fill_between(
        x[mask],
        y_min,
        y_max,
        color="red",
        alpha=0.2,
    )  # , label='Exterior')

    # label the interior and exterior regions
    plt.text(
        x_boundary + 0.5,
        0.5,
        "Outside Training Bounds",
        color="black",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="w"),
    )
    plt.text(
        x_boundary - 1.5,
        0.5,
        "Inside Training Bounds",
        color="black",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="w"),
    )

    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])
    plt.xlabel("Radius [R]")
    plt.ylabel("Model Weight [-]")

    plt.legend()

    vis.save(plt.gcf(), "BoundaryCondition")

    plt.show()


if __name__ == "__main__":
    main()
