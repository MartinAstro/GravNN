import numpy as np
from experiment_setup import setup_experiments
from interfaces import select_model

from GravNN.Networks.Configs import *
from GravNN.Networks.utils import populate_config_objects
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer


def get_planes_metrics(exp):
    percent_error = np.nanmean(exp.percent_error_acc)
    rms_error = np.nanmean(exp.RMS_acc)
    return {
        "percent_planes": percent_error,
        "rms_planes": rms_error,
    }


def get_extrap_metrics(exp):
    # Interior, Exterior, Extrapolation
    vis = ExtrapolationVisualizer(exp)

    x = vis.x_test
    x_interpolation = x[: vis.max_idx]
    interior_mask = x_interpolation < 1.0

    y_interpolation = vis.experiment.losses["percent"][vis.idx_test][: vis.max_idx]
    y_extrapolation = vis.experiment.losses["percent"][vis.idx_test][vis.max_idx :]

    y_interior = y_interpolation[interior_mask]
    y_exterior = y_interpolation[~interior_mask]

    percent_interior = np.nanmean(y_interior)
    percent_exterior = np.nanmean(y_exterior)
    percent_extrapolation = np.nanmean(y_extrapolation)

    metrics = {
        "percent_interior": percent_interior,
        "percent_exterior": percent_exterior,
        "percent_extrapolation": percent_extrapolation,
    }
    return metrics


def get_traj_metrics(exp):
    test_model = exp.test_models[0]
    pos_error = test_model.metrics["pos_diff"][-1]
    state_error = test_model.metrics["state_diff"][-1]
    dt = test_model.orbit.elapsed_time[-1]

    return {
        "pos_error": pos_error,
        "state_error": state_error,
        "dt": dt,
    }


def extract_metrics(model):
    metrics = {}
    metrics.update(get_planes_metrics(model.plane_exp))
    metrics.update(get_extrap_metrics(model.extrap_exp))
    metrics.update(get_traj_metrics(model.trajectory_exp))
    return metrics


def load_experiment(experiment, config):
    model = select_model(experiment["model_name"][0])
    model.configure(config)
    model.load()
    model.evaluate()
    return model


def main():
    experiments = setup_experiments()

    for idx, exp in enumerate(experiments):
        config = get_default_eros_config()
        config.update(PINN_III())
        config.update(ReduceLrOnPlateauConfig())
        config.update(exp)
        config = populate_config_objects(config)
        config["comparison_idx"] = [idx]

        model = load_experiment(exp, config)
        metrics = extract_metrics(model)
        print(metrics)


if __name__ == "__main__":
    main()
