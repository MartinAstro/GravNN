import os
import pickle
import time

import pandas as pd
import tensorflow as tf

import GravNN
from GravNN.Networks import utils


def count_nonzero_params(model):
    params = 0
    for v in model.trainable_variables:
        params += tf.math.count_nonzero(v)
    return params.numpy()


class ModelSaver:
    def __init__(
        self,
        model,
        history=None,
        save_dir=os.path.dirname(GravNN.__file__) + "/../Data",
    ):
        self.config = model.config
        self.network = model.network
        self.history = history
        self.save_dir = save_dir

        # saving

    def model_size_stats(self):
        """Method which computes the number of trainable variables in the model as well
        as the binary size of the saved network and adds it to the configuration
        dictionary.
        """
        size_stats = {
            "params": [count_nonzero_params(self.network)],
            "size": [utils.get_gzipped_model_size(self)],
        }
        self.config.update(size_stats)

    def save(self, df_file=None):
        """Add remaining training / model variables into the configuration dictionary,
        then save the config variables into its own pickled file, and potentially add
        it to an existing dataframe defined by `df_file`.

        Args:
            df_file (str or pd.Dataframe, optional): path to dataframe to which the
            config variables should be appended or the loaded dataframe itself.
            Defaults to None.
        """
        self.model_size_stats()

        # ensure that the id is unique by using pid in id
        model_id = (
            pd.Timestamp(time.time(), unit="s").to_julian_date() + 1 / os.getpid()
        )
        self.config["id"] = [model_id]
        self.config["timetag"] = [model_id]

        # Save the network
        os.makedirs(f"{self.save_dir}/Dataframes/", exist_ok=True)
        os.makedirs(f"{self.save_dir}/Networks/", exist_ok=True)
        network_dir = f"{self.save_dir}/Networks/{model_id}/"
        self.network.save(network_dir + "network")
        self.network.save_weights(network_dir + "weights")

        # save the training history + delete to reduce memory
        if self.history is not None:
            with open(network_dir + "history.data", "wb") as f:
                pickle.dump(self.history.history, f)
            del self.history

        # convert configuration info to dataframe + save
        config = dict(sorted(self.config.items(), key=lambda kv: kv[0]))
        df = pd.DataFrame().from_dict(config).set_index("timetag")
        df.to_pickle(network_dir + "config.data")

        # concatenate config to preexisting dataframe if requested
        if df_file is not None:
            utils.save_df_row(self.config, f"{self.save_dir}/Dataframes/{df_file}")

        return network_dir
