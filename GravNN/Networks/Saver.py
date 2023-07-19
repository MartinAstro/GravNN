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
    ):
        self.config = model.config
        self.network = model.network
        self.history = history

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

    def extract_save_directory(self, df_file):
        # assume save dir is GravNN/Data
        save_dir = os.path.dirname(GravNN.__file__) + "/../Data"

        # unless there is a /Data/ dir that is within the df_path
        if (type(df_file) == str) and os.path.isabs(df_file):
            save_dir = os.path.dirname(df_file)

            # if the filepath has Data/ in it, clip everything past it
            # and just save into Data/Dataframes and Data/Networks...
            if "/Data/" in df_file:
                save_dir = df_file.split("/Data/")[0] + "/Data"

        self.save_dir = save_dir
        self.config["save_dir"] = [save_dir]

    def extract_basename(self, df_file):
        basename = ""
        if type(df_file) == str:
            basename = os.path.basename(df_file)
        return basename

    def assign_model_id(self):
        # ensure that the id is unique by using pid in
        model_id = (
            pd.Timestamp(time.time(), unit="s").to_julian_date() + 1 / os.getpid()
        )
        self.config["id"] = [model_id]
        self.config["timetag"] = [model_id]
        return model_id

    def save_network(self, model_id):
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

        self.network_dir = network_dir

    def save_dataframe(self, df_file):
        if df_file is not None:
            # concatenate config to preexisting dataframe if requested
            basename = self.extract_basename(df_file)
            os.makedirs(f"{self.save_dir}/Dataframes/", exist_ok=True)
            df_save_path = f"{self.save_dir}/Dataframes/{basename}"
            utils.save_df_row(
                self.config,
                df_save_path,
            )

    def save(self, df_file=None):
        """Add remaining training / model variables into the configuration dictionary,
        then save the config variables into its own pickled file, and potentially add
        it to an existing dataframe defined by `df_file`.

        Args:
            df_file (str or pd.Dataframe, optional): path to dataframe to which the
            config variables should be appended or the loaded dataframe itself.
        """
        self.extract_save_directory(df_file)
        self.model_size_stats()
        model_id = self.assign_model_id()
        self.save_network(model_id)
        self.save_dataframe(df_file)

        return self.network_dir
