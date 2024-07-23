import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Visualization.VisualizationBase import VisualizationBase


def make_potential(x, e):
    def H(x, k, r):
        return (1 + np.tanh(k * (x - r))) / 2

    r_inv_cap = np.clip(1 / x, 0, 1)
    r_cap = np.clip(x, 0, 1)

    a = 1
    u_pm_external = r_inv_cap
    u_external_full = -u_pm_external

    # Internal
    u_external_pm_boundary = 1 / a

    u_boundary = -u_external_pm_boundary
    u_internal = 1 * (r_cap**2 / a**3) + 2 * u_boundary

    u_analytic = np.where(x < a, u_internal, u_external_full)

    # decrease the weight of the model in the region between
    # the interior of the asteroid and out to r < 1 + e, where
    # e is the eccentricity of the asteroid geometry, because
    # in this regime, a point mass / SH assumption adds unnecessary
    # error.
    r_external = 1 + e
    k_external = 0.5
    h_external = H(x, k_external, r_external)
    u_w_analytic = u_analytic * h_external

    return u_w_analytic, u_analytic, h_external


def main():
    vis = VisualizationBase()

    x_min, x_max = 0.3, 100
    x = np.linspace(x_min, x_max, 1000)

    a = Eros().radius
    b = Eros().radius_min
    e = np.sqrt(1 - b**2 / a**2)

    U_analytic, U_pm, w = make_potential(x, e)

    vis.newFig()
    plt.plot(
        x,
        U_pm,
        color="black",
        linewidth=1.5,
        label=r"$U_{\text{LF}}$: Analytic",
    )  # , label='$G(x)$')
    plt.plot(
        x,
        U_analytic,
        color="red",
        linewidth=1.5,
        label=r"$\hat{U}_{\text{LF}}$: Weighted Analytic",
    )  # , label='$G(x)$')
    plt.legend()

    # Shade the region where r < 1
    y_max = 0.5
    y_min = -2
    # plt.fill_between(x[mask_inv], y_min, y_max, color='green', alpha=0.2) #, label='Interior')
    # plt.fill_between(x[mask], y_min, y_max, color='red', alpha=0.2) #, label='Exterior')

    # label the interior and exterior regions
    # plt.text(x_boundary + 0.5, 0.5, 'Outside Training Bounds', color='black', fontsize=8)
    # plt.text(x_boundary - 1.5, 0.5, 'Inside Training Bounds', color='black', fontsize=8)

    plt.vlines(1 + e, y_min, y_max, colors="grey")
    plt.annotate(
        r"$r^{\star}=1+e$",
        ((1 + e) + 0.5, y_min + 0.15),
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="gray", alpha=0.5),
    )
    # plt.annotate(r'$r^{\star}=1+e$', ((1+e)-0.15, y_min/2-0.20), rotation=90, ha='center', va='bottom', fontsize=8)

    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])
    plt.xlabel("Radius [R]")
    plt.ylabel("Potential [-]")
    plt.legend(loc="upper left")

    # plt.xscale('log')
    plt.twinx()
    plt.plot(
        x,
        w,
        color="blue",
        linewidth=1,
        linestyle="--",
        label=r"$w_{\text{LF}}$",
    )  # , label='$H(x)$')
    plt.ylabel("Weight [-]")
    plt.legend(loc="lower right")
    plt.grid(False)

    plt.xscale("log")

    vis.save(plt.gcf(), "FuseModel")

    plt.show()


if __name__ == "__main__":
    main()
