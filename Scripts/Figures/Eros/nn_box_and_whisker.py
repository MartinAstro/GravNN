import pandas as pd
import plotly.io as pio

pio.kaleido.scope.mathjax = None
import glob

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Trajectories import RandomAsteroidDist, SurfaceDist

planet = Eros()
model_file = planet.obj_200k
interior_bound = planet.radius
exterior_bound = planet.radius * 3

pinn_df = pd.read_pickle("Data/Dataframes/eros_official_w_noise.data")
transformer_df = pd.read_pickle(
    "Data/Dataframes/eros_official_noise_transformer_no_annealing.data",
)
sh_models = glob.glob(
    "GravNN/Files/GravityModels/Regressed/Eros/RandomAsteroidDist/BLLS/**/**/**/**.csv",
)


def get_file_info(model_path):
    directories = model_path.split("/")
    directories = directories[-4:]
    N = int(directories[0].split("_")[1])
    M = int(directories[1].split("_")[1])
    N_data = int(directories[2])
    entries = directories[3].split("_")[1].split(".")
    noise = float(entries[0] + "." + entries[1])
    return N, M, N_data, noise


def compute_stats(model, trajectory, config):
    # Compute the true error of the 20,000 point distributions (or 200k if surface)
    x, a, u = get_poly_data(trajectory, model_file, **config)

    # if the distribution is the surface, sample only 20000 points
    indices = np.linspace(0, len(x) - 1, len(x))
    sample_idx = np.random.choice(indices, 20000).astype(int)
    x_sample = x[sample_idx]
    a_sample = a[sample_idx]

    # compute error
    a_pred = model.compute_acceleration(x_sample.astype(np.float32))
    percent = (
        np.linalg.norm(a_sample - a_pred, axis=1)
        / np.linalg.norm(a_sample, axis=1)
        * 100
    )

    return percent


def compute_sh_stats(model, trajectory):
    # Compute the true error of the 20,000 point distributions (or 200k if surface)
    x, a_true, u = get_poly_data(trajectory, model_file, point_mass_removed=[False])
    N, M, N_data, noise = get_file_info(model)
    x, a_regress, u = get_sh_data(
        trajectory,
        model,
        max_deg=N,
        deg_removed=-1,
        override=[True],
    )
    percent = (
        np.linalg.norm(a_regress - a_true, axis=1)
        / np.linalg.norm(a_true, axis=1)
        * 100
    )
    return percent


def compute_surface_stats(model, config=None):
    trajectory = SurfaceDist(planet, model_file)
    if config is None:
        stats = compute_sh_stats(model, trajectory)
    else:
        stats = compute_stats(model, trajectory, config)
    return stats


def compute_interior_stats(model, config=None):
    trajectory = RandomAsteroidDist(planet, [0, interior_bound], 20000, model_file)
    if config is None:
        stats = compute_sh_stats(model, trajectory)
    else:
        stats = compute_stats(model, trajectory, config)
    return stats


def compute_exterior_stats(model, config=None):
    trajectory = RandomAsteroidDist(
        planet,
        [interior_bound, exterior_bound],
        20000,
        model_file,
    )
    if config is None:
        stats = compute_sh_stats(model, trajectory)
    else:
        stats = compute_stats(model, trajectory, config)
    return stats


def load_model(identifier):
    if identifier in pinn_df["id"].values:
        config, model = load_config_and_model(pinn_df, identifier)
    elif identifier in transformer_df["id"].values:
        config, model = load_config_and_model(transformer_df, identifier)
    else:
        config, model = None, identifier
    return config, model


def get_color(model):
    return {
        "PINN 00": "rgba(83, 64, 50, 0.5)",
        "PINN A": "rgba(103, 64, 100, 0.5)",
        "PINN AP": "rgba(123, 64, 150, 0.5)",
        "PINN ALC": "rgba(143, 64, 200, 0.5)",
        "PINN APLC": "rgba(163, 64, 250, 0.5)",
        "PIT 00": "rgba(13, 50, 104, 0.5)",
        "PIT A": "rgba(13, 100, 84, 0.5)",
        "PIT AP": "rgba(13, 150, 64, 0.5)",
        "PIT ALC": "rgba(13, 200, 44, 0.5)",
        "PIT APLC": "rgba(13, 250, 24, 0.5)",
        "SH 4": "rgba(153, 50, 14, 0.5)",
        "SH 8": "rgba(203, 30, 14, 0.5)",
        "SH 16": "rgba(253, 10, 14, 0.5)",
    }[model]


def get_stat(distribution):
    return {
        "exterior": compute_exterior_stats,
        "interior": compute_interior_stats,
        "surface": compute_surface_stats,
    }[distribution]


def get_legend_group(model):
    return {
        "PINN 00": "PINN",
        "PINN A": "PINN",
        "PINN AP": "PINN",
        "PINN ALC": "PINN",
        "PINN APLC": "PINN",
        "PIT 00": "PIT",
        "PIT A": "PIT",
        "PIT AP": "PIT",
        "PIT ALC": "PIT",
        "PIT APLC": "PIT",
        "SH 4": "SH",
        "SH 8": "SH",
        "SH 16": "SH",
    }[model]


def add_lines(fig):
    fig.add_hline(
        1,
        line=go.layout.shape.Line(color="green"),
        opacity=1,
        line_width=0.7,
    )
    fig.add_hline(
        10,
        line=go.layout.shape.Line(color="yellow"),
        opacity=1,
        line_width=0.7,
    )
    fig.add_hline(
        100,
        line=go.layout.shape.Line(color="red"),
        opacity=1,
        line_width=0.7,
    )
    return fig


def main():
    df = pd.read_pickle("Data/Dataframes/box_and_whisker.data")

    # distribution = "exterior" # "interior", "surface"
    # fig = go.Figure()
    # model_types = np.unique(sub_df.index.get_level_values(2).to_numpy())
    # x = sub_df.index.get_level_values(1).to_numpy() # Groups the box-plots into natural clusters
    for distribution in ["exterior", "interior", "surface"]:
        # for distribution in ['exterior']:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            subplot_titles=("Noise: 0%", "Noise: 10%", "Noise: 20%"),
        )
        sub_df = df.loc[0.0]
        num_models = len(sub_df)
        step = 1
        for i in range(0, num_models, step):
            row = sub_df.iloc[i]
            model_name = row.name[1].replace("Transformer", "PIT")
            showlegend = True if i < 13 else False
            config, model = load_model(row[0])
            percent_error = get_stat(distribution)(model, config)
            fig.add_trace(
                go.Box(
                    y=percent_error,
                    x=np.repeat("N=" + str(row.name[0]), len(percent_error)),
                    legendgroup=get_legend_group(model_name),
                    name=model_name,
                    offsetgroup=str(model_name),
                    marker_color=get_color(model_name),
                    marker_size=1,
                    # marker_line_width=1,
                    line_width=0.5,
                    opacity=1,
                    width=0,
                    notchwidth=0,
                    showlegend=showlegend,
                ),
                row=1,
                col=1,
            )
            print(i)
        fig = add_lines(fig)

        sub_df = df.loc[0.1]
        for i in range(0, num_models, step):
            row = sub_df.iloc[i]
            model_name = row.name[1].replace("Transformer", "PIT")
            showlegend = True if i < 13 else False
            config, model = load_model(row[0])
            percent_error = get_stat(distribution)(model, config)
            fig.add_trace(
                go.Box(
                    y=percent_error,
                    x=np.repeat("N=" + str(row.name[0]), len(percent_error)),
                    legendgroup=get_legend_group(model_name),
                    name=model_name,
                    offsetgroup=str(model_name),
                    marker_color=get_color(model_name),
                    marker_size=1,
                    # marker_line_width=1,
                    line_width=0.5,
                    opacity=1,
                    width=0,
                    notchwidth=0,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            print(i)
        fig = add_lines(fig)

        sub_df = df.loc[0.2]
        for i in range(0, num_models, step):
            row = sub_df.iloc[i]
            model_name = row.name[1].replace("Transformer", "PIT")
            showlegend = True if i < 13 else False
            config, model = load_model(row[0])
            percent_error = get_stat(distribution)(model, config)
            fig.add_trace(
                go.Box(
                    y=percent_error,
                    x=np.repeat("N=" + str(row.name[0]), len(percent_error)),
                    legendgroup=get_legend_group(model_name),
                    name=model_name,
                    offsetgroup=str(model_name),
                    marker_color=get_color(model_name),
                    marker_size=1,
                    # marker_line_width=1,
                    line_width=0.5,
                    opacity=1,
                    width=0,
                    notchwidth=0,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            print(i)
        fig = add_lines(fig)

        fig.update_yaxes(
            {
                "type": "log",
                "linecolor": "black",
                "ticks": "outside",
                "gridcolor": "LightGray",
                "title": r"Acceleration Error [%]",
                "range": [-1, 2],
                "dtick": "D1",
                "tickvals": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    20,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    100,
                ],
                "ticktext": [
                    "0.1",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "1",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "10",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "100",
                ],
            },
        )
        fig.update_xaxes({"zerolinecolor": "black", "title": "Data Samples"})
        # update global parameters
        fig.update_layout(
            boxmode="group",  # group together boxes of the different traces for each value of x
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            boxgroupgap=0.1,
            # boxgap=0.1
        )

        fig.update_layout(
            autosize=True,
            width=600,
            height=900,
            template="none",
            font={
                "family": "serif",
            },
        )

        print("Plotting Image")

        # if distribution == 'surface':
        #     app = dash.Dash()
        #     app.layout = html.Div([dcc.Graph(figure=fig, config={'staticPlot':True})])
        #     app.run_server(debug=False, use_reloader=False)
        # else:
        # fig.write_image("Plots/Asteroid/box_and_whisker_"+ distribution + "_v2.pdf")
        fig.write_image("Plots/Asteroid/box_and_whisker_" + distribution + "_v2.jpeg")
        fig.write_image("Plots/Asteroid/box_and_whisker_" + distribution + "_v2.pdf")
    fig.show()


if __name__ == "__main__":
    main()
