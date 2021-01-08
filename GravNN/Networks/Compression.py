import tensorflow as tf
import tensorflow_model_optimization as tfmot
from GravNN.Networks.Model import CustomModel
from GravNN.Networks.Callbacks import CustomCallback

def prune_model(model, dataset, val_dataset, config):
    history = None
    if config['sparsity'][0] is not None:
        #Pruning https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                                initial_sparsity=0.00,
                                final_sparsity=config['sparsity'][0],
                                begin_step=0,
                                end_step=config['fine_tuning_epochs'][0])
        # pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(
        #                         target_sparsity=config['sparsity'][0],
        #                         begin_step=0,
        #                         frequency=10)
        pruned_network = tfmot.sparsity.keras.prune_low_magnitude(model.network, pruning_schedule=pruning_schedule)
        pruned_model = CustomModel(config, pruned_network)
        pruned_model.compile(loss='mse', optimizer='adam')

        callback = CustomCallback()
        history = pruned_model.fit(dataset, 
                    validation_data=val_dataset,
                    epochs=config['fine_tuning_epochs'][0], 
                    verbose=0,
                    callbacks=[callback,
                                tfmot.sparsity.keras.UpdatePruningStep()])
        history.history['time_delta'] = callback.time_delta
        pruned_network = tfmot.sparsity.keras.strip_pruning(pruned_model.network)
        pruned_model = CustomModel(config, pruned_network)
        model = pruned_model
    return model, history 


def cluster_model(model, dataset, val_dataset, config):
    history = None
    if config['num_w_clusters'][0] is not None:
        #Weight Clustering: https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html
        clustering_params = {
            'number_of_clusters': config['num_w_clusters'][0],
            'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR
        }

        clustered_network = tfmot.clustering.keras.cluster_weights(model.network, **clustering_params)
        clustered_model = CustomModel(config, clustered_network)
        clustered_model.compile(loss='mse', optimizer='adam')

        callback = CustomCallback()
        history = clustered_model.fit(dataset, 
                            validation_data=val_dataset,
                            epochs=config['fine_tuning_epochs'][0], 
                            verbose=0,
                            callbacks=[callback])
        history.history['time_delta'] = callback.time_delta
        clustered_network = tfmot.clustering.keras.strip_clustering(clustered_model.network)
        model = CustomModel(config, clustered_network)
    return model, history 


def quantize_model(model, dataset, val_dataset, config):
    history = None
    if config['quantization'][0] is not None:
        # Quantizations: https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html
        qunatized_network = tfmot.quantization.keras.quantize_model(model.network)
        model = CustomMoel(config, quantized_network)

    return model, history 
