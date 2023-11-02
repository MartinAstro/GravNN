from experiment_setup import setup_experiments
from interfaces import select_model

from GravNN.Networks.Configs import *
from GravNN.Networks.Data import DataSet
from GravNN.Networks.utils import populate_config_objects


def run(experiment, idx):
    config = get_default_eros_config()
    # config['N_dist'] = [1000]
    # config["radius_min"] = [Eros().radius * 10]
    # config["radius_max"] = [Eros().radius * 15]
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    config.update(experiment)
    config = populate_config_objects(config)
    config["comparison_idx"] = [idx]

    data = DataSet(config)

    model_name = config["model_name"][0]
    wrapper = select_model(model_name)
    wrapper.configure(config)
    wrapper.train(data)
    wrapper.save()
    wrapper.evaluate(override=True)


def main():
    experiments = setup_experiments()

    for idx, exp in enumerate(experiments):
        run(exp, idx)


if __name__ == "__main__":
    main()
