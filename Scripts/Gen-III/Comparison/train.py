from experiment_setup import *
from extract_metrics import extract_metrics, save_metrics
from interfaces import select_model

from GravNN.Networks.Configs import *
from GravNN.Networks.Data import DataSet
from GravNN.Networks.utils import populate_config_objects


def run(experiment, idx):
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
    wrapper.evaluate(override=True)

    metrics = extract_metrics(wrapper)
    save_metrics(metrics, idx)


def main():
    experiments = setup_experiments()
    for idx, exp in enumerate(experiments):
        print(exp)
        run(exp, idx)


if __name__ == "__main__":
    main()
