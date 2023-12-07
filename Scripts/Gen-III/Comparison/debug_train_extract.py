from experiment_setup import *
from extract_metrics import extract_metrics
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

    # Does this work?
    extract_metrics(wrapper)
    # save_metrics(metrics, idx)


def main():
    from extract_metrics import extract_metrics, load_experiment

    experiments = setup_fast_experiments()

    # The Run Part
    for idx, exp in enumerate(experiments):
        print(exp)
        run(exp, idx)

        # The Extract Part
        experiments = setup_experiments()
        exp = experiments[idx]
        model_name = exp["model_name"][0]
        config = get_default_config(model_name)
        config.update(exp)
        config = populate_config_objects(config)

        config["comparison_idx"] = [idx]

        model = load_experiment(exp, config)
        metrics = extract_metrics(model)
        metrics.update(exp)
        metrics.update({"model_name": model_name})


if __name__ == "__main__":
    main()
