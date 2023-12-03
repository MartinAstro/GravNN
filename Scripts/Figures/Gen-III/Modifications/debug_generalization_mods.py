import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Analysis.SurfaceExperiment import SurfaceExperiment
from GravNN.GravityModels.HeterogeneousPoly import (
    generate_heterogeneous_model,
    get_hetero_poly_data,
)
from GravNN.Networks.Configs import *
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Visualization.HistoryVisualizer import HistoryVisualizer
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.Visualization.SurfaceVisualizer import SurfaceVisualizer

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


df_file = "Data/Dataframes/eros_cost_fcn_percent_mse_test.data"
df_file = "Data/Dataframes/eros_cost_fcn_percent_test.data"

df_file = "Data/Dataframes/eros_cost_fcn_percent_test_SB.data"
df_file = "Data/Dataframes/eros_cost_fcn_percent_mse_test_SB.data"
df_file = "Data/Dataframes/eros_cost_fcn_percent_mse_test_SB_no_BC.data"
df_file = "Data/Dataframes/eros_cost_fcn_pinn_III_SB_no_fuse_or_BC.data"
df_file = "Data/Dataframes/eros_cost_fcn_pinn_III_SB_no_fuse_or_BC_or_scaling.data"
df_file = (
    "Data/Dataframes/eros_cost_fcn_pinn_III_SB_no_fuse_or_BC_or_scaling_or_percent.data"
)
df_file = "Data/Dataframes/eros_cost_fcn_pinn_III_SB_no_fuse_or_BC_or_scaling_or_percent_higher_clip.data"

df_file = "Data/Dataframes/eros_cost_fcn_pinn_II.data"
# df_file = "Data/Dataframes/eros_cost_fcn_pinn_III_MSE_features.data" # Very low
df_file = "Data/Dataframes/eros_cost_fcn_pinn_III_scaling_features.data"  # Very low
df_file = "Data/Dataframes/eros_cost_fcn_pinn_III_fuse.data"  # Very low


def main():
    config = get_default_eros_config()
    config.update(PINN_III())
    # config.update(PINN_II())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [50000],
        "N_train": [2**13],
        "N_val": [4096],
        "num_units": [16],
        "layers": [[3, 1, 1, 1, 1, 1, 1, 3]],
        "batch_size": [2**11],
        "epochs": [10000],
        "gravity_data_fcn": [get_hetero_poly_data],
        "PINN_constraint_fcn": ["pinn_a"],
        "loss_fcns": [["rms", "percent"]],
        "fuse_models": [True],
        "enforce_bc": [True],
        "scale_nn_potential": [True],
        "decay_rate": [0.5],
        "min_delta": [0.001],
        # "jit_compile": [False],
        # "eager": [True],
        "tanh_k": [1.0],
        # "tanh_r" : [3],
    }
    args = configure_run_args(config, hparams)

    configs = [run(*args[0])]
    save_training(df_file, configs)


def plot():
    df_file_list = [
        # "Data/Dataframes/eros_cost_fcn_percent_test_SB.data",
        # "Data/Dataframes/eros_cost_fcn_percent_mse_test_SB.data",
        # "Data/Dataframes/eros_cost_fcn_percent_mse_test_SB_no_BC.data",
        # "Data/Dataframes/eros_cost_fcn_pinn_II.data",
        # "Data/Dataframes/eros_cost_fcn_pinn_III_SB_no_fuse_or_BC.data",
        # "Data/Dataframes/eros_cost_fcn_pinn_III_SB_no_fuse_or_BC_or_scaling.data",
        # "Data/Dataframes/eros_cost_fcn_pinn_III_SB_no_fuse_or_BC_or_scaling_or_percent.data",
        # "Data/Dataframes/eros_cost_fcn_pinn_III_SB_no_fuse_or_BC_or_scaling_or_percent_higher_clip.data",
        # "Data/Dataframes/eros_cost_fcn_pinn_III_MSE_features.data",
        "Data/Dataframes/eros_cost_fcn_pinn_III_scaling_features.data",
        "Data/Dataframes/eros_cost_fcn_pinn_III_fuse.data",
        # "Data/Dataframes/eros_cost_fcn_pinn_III_MSE_Percent_features.data",
        # "Data/Dataframes/eros_cost_fcn_pinn_III_MSE_Percent_features_SB.data",
    ]
    for df_file in df_file_list:
        config, model = load_config_and_model(df_file, idx=-1)
        experiment = PlanesExperiment(
            model,
            config,
            [-Eros().radius * 3, Eros().radius * 3],
            50,
        )
        experiment.run()
        visualizer = PlanesVisualizer(experiment)
        visualizer.plot(z_max=10)

        experiment = ExtrapolationExperiment(model, config, points=1000)
        experiment.run()
        visualizer = ExtrapolationVisualizer(experiment)
        visualizer.plot_interpolation_percent_error()
        plt.yscale("log")
        visualizer.plot_extrapolation_percent_error()
        plt.yscale("log")

        # Also do surface experiment
        true_model = generate_heterogeneous_model(Eros(), config["obj_file"][0])
        experiment = SurfaceExperiment(model, true_model)
        experiment.run()
        print(np.mean(experiment.percent_error_acc))
        visualizer = SurfaceVisualizer(experiment)
        # visualizer.plot_percent_error(max_percent=0.35)

        visualizer = HistoryVisualizer(model, config)
        visualizer.plot_loss(log_y=True)

    plt.show()


def run(config):
    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.Saver import ModelSaver
    from GravNN.Networks.utils import configure_tensorflow, populate_config_objects

    configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    pprint(config)

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    model = PINNGravityModel(config)
    history = model.train(data)
    model.config["val_loss"] = history.history["val_percent_mean"][-1]

    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
    plot()
