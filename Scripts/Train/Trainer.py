import logging
import multiprocessing as mp
from pprint import pprint

from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import permutate_dict


class Trainer:
    def __init__(self, config, hparams, df_file):
        self.config = config
        self.df_file = df_file
        for key, value in hparams.items():
            self.config[key] = [value]

    def run(self):
        from GravNN.Networks.Data import DataSet
        from GravNN.Networks.Model import PINNGravityModel
        from GravNN.Networks.Saver import ModelSaver
        from GravNN.Networks.utils import configure_tensorflow, populate_config_objects

        configure_tensorflow(self.config)

        # Standardize Configuration
        config = populate_config_objects(self.config)
        pprint(config)

        # Get data, network, optimizer, and generate model
        data = DataSet(config)
        model = PINNGravityModel(config)
        history = model.train(data)

        saver = ModelSaver(model, history)
        saver.save(df_file=None)

        print(f"Model ID: [{model.config['id']}]")
        return model.config

    def save(self, data):
        save_training(self.df_file, [data])


class PoolTrainer:
    def __init__(self, config, hparams, df_file):
        self.df_file = df_file

        # permute all hparams
        hparams_permutations = permutate_dict(hparams)

        # Initialize their trainers
        self.trainers = []
        for hparams in hparams_permutations:
            self.trainers.append(Trainer(config, hparams, df_file))

    def run(self, threads=1):
        def trainer_run(trainer):
            return trainer.run()

        if threads == 1:
            logging.info("Running in serial")
            results = []
            for trainer in self.trainers:
                result = trainer.run()
                results.append(result)
        else:
            with mp.Pool(threads) as pool:
                results = pool.starmap_async(trainer_run, self.trainers)
                results.get()

        save_training(self.df_file, results)
