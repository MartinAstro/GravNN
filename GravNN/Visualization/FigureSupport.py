import matplotlib.pyplot as plt
import numpy as np


def get_vlim_bounds(dist, sigma):
    mu = np.mean(dist)
    std = np.std(dist)
    vlim_min = clamp(mu - sigma * std, 0, np.inf)
    vlim_max = mu + sigma * std
    return [vlim_min, vlim_max]


def clamp(x, smallest, largest):
    return max(smallest, min(x, largest))


def format_potential_as_Nx3(u):
    U_Nx3 = np.zeros((len(u), 3))
    try:
        U_Nx3[:, 0] = u[:, 0]
    except Exception:
        U_Nx3[:, 0] = u
    return U_Nx3


def sh_pareto_curve(
    sh_df,
    log=True,
    sigma=2,
    metric="mean",
    label="MRSE",
):
    plot_fcn = plt.semilogx if log else plt.plot

    params = sh_df.index * (sh_df.index + 1)
    sh_rse = sh_df["rse_" + metric]
    sh_sigma = sh_df["sigma_" + str(sigma) + "_" + metric]
    sh_sigma_c = sh_df["sigma_" + str(sigma) + "_c_" + metric]

    plot_fcn(params, sh_rse, label=label + r"($\mathcal{A}$)")
    plot_fcn(params, sh_sigma, label=label + r"($\mathcal{F}$)")
    plot_fcn(params, sh_sigma_c, label=label + r"($\mathcal{C}$)")

    ax = plt.gca()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    plt.ylabel("MRSE [m/s$^2$]")
    plt.xlabel("Parameters")


def nn_pareto_curve(
    df,
    orbit_name,
    linestyle=None,
    marker=None,
    log=True,
    sigma=2,
    metric="mean",
):
    plt.gca().set_prop_cycle(None)

    plot_fcn = plt.semilogx if log else plt.plot

    params = df.params
    df_rse = df[orbit_name + "_rse_" + metric]
    df_sigma = df[orbit_name + "_sigma_" + str(sigma) + "_" + metric]
    df_sigma_c = df[orbit_name + "_sigma_" + str(sigma) + "_c_" + metric]

    plot_fcn(params, df_rse, linestyle=linestyle, marker=marker)
    plot_fcn(params, df_sigma, linestyle=linestyle, marker=marker)
    plot_fcn(params, df_sigma_c, linestyle=linestyle, marker=marker)
    plt.legend()
