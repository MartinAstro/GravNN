import sys

from experiment_setup import *
from interfaces import select_model

from GravNN.Networks.Configs import *
from GravNN.Networks.Data import DataSet
from GravNN.Networks.utils import populate_config_objects


def run(experiment, idx):
    print(experiment)
    model_name = experiment["model_name"][0]

    config = get_default_config(model_name)
    config.update(experiment)
    config = populate_config_objects(config)
    config["comparison_idx"] = [idx]

    data = DataSet(config)

    wrapper = select_model(model_name)
    wrapper.configure(config)
    wrapper.train(data)
    wrapper.save()
    wrapper.evaluate(override=False)


def main():
    experiments = setup_experiments()
    idx = int(sys.argv[1])
    exp = experiments[idx]
    run(exp, idx)


if __name__ == "__main__":
    main()
