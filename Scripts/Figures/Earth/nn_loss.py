# %%
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d

from GravNN.Networks.Networks import ResNet

nn_df = pd.read_pickle(
    "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\Data\\Dataframes\\traditional_nn_df.data",
)
pinn_df = pd.read_pickle(
    "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\Data\\Dataframes\\pinn_df.data",
)

# df = pd.concat([nn_df, pinn_df])
df = nn_df

# %%
backup = copy.deepcopy(df)


# %%
loss_list = []
val_loss_list = []

for history in df["history"]:
    loss_list.append(history["loss"][-1])
    val_loss_list.append(history["val_loss"][-1])
df["loss"] = loss_list
df["val_loss"] = val_loss_list

networks = df["network_type"] == ResNet
networks.replace(True, "resnet", inplace=True)
networks.replace(False, "traditional", inplace=True)
df["network_type"] = networks

df["model_param_est"] = df["Brillouin_param_rse_median"] * (
    df["Brillouin_param_rse_median"] + 1
)
sub_df = df[
    [
        "params",
        "batch_size",
        "N_train",
        "activation",
        "initializer",
        "learning_rate",
        "network_type",
        "loss",
        "val_loss",
    ]
].sort_values(
    by="val_loss",
    ascending=False,
)  # , 'Brillouin_sh_diff_median']

# %%


# %%

plt.figure(figsize=(15, 20))
for i in range(0, len(df)):
    row = df.iloc[i]
    y = row["history"]["val_loss"]
    x = np.linspace(0, 50000, len(y))

    N = 1000
    N_end = 99900  # None

    y_hat = y[N:N_end]
    x_hat = x[N:N_end]

    poly_y = uniform_filter1d(y_hat, 100)

    bias = 0  # np.mean(poly_y)
    plt.semilogy(
        poly_y - bias,
        alpha=1,
        label=str(row["decay_rate_epoch"]) + "_" + str(row["learning_rate"]),
    )
    # plt.plot(poly_y-bias,alpha=1, label=str(row['num_units']))

plt.legend()


# %%
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d

plt.figure(figsize=(15, 20))
for i in range(0, len(df)):
    row = df.iloc[i]
    y = row["history"]["val_loss"]
    x = np.linspace(0, 50000, len(y))

    N = 1000
    N_end = 99900
    y_hat = y[N:N_end]
    x_hat = x[N:N_end]

    poly_y = uniform_filter1d(y_hat, 5000)

    grad = np.gradient(poly_y, x_hat)
    grad_poly_y = uniform_filter1d(grad, 5000)

    plt.plot(grad_poly_y, alpha=1)
    plt.gca().set_ylim([-1e-8, 0.1e-8])

# plt.hline
# %%
