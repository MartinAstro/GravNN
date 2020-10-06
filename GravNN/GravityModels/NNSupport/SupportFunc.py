import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict
import matplotlib.lines as mlines
cmaps = OrderedDict()
cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
def plot_metrics(histories, save_location=None):
    plt.figure()
    colors = plt.cm.tab20(np.linspace(0,1,len(histories)))
    color_handles=[]
    for i in range(len(histories)): 
        history = histories[i]
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        a = plt.plot(epochs, loss, linestyle='-',  c=colors[i])
        plt.plot(epochs, val_loss, linestyle='--', c=colors[i])
        color_handles.append(mlines.Line2D([], [], color=colors[i], label="Model: " + str(i)))
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    training_line = mlines.Line2D([], [], linestyle='-', color='black', label='Training')
    validation_line = mlines.Line2D([], [], linestyle='--', color='black', label='Validation')

    first_legend = plt.legend(handles=[training_line, validation_line], loc='upper left')
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=color_handles, loc='upper right')
    if save_location is not None:
        os.makedirs(save_location, exist_ok=True)
        plt.savefig(plt.gcf(), save_location + "loss.pdf", bbox_inches='tight')
    return

def calc_metrics(meas, truth):
    error_inst = abs(np.divide((meas - truth),truth)*100)
    mse = np.square(meas - truth).mean(axis=None)
    rmse = np.sqrt(mse).mean(axis=None)
    print("Avg: " + str(np.average(error_inst)))
    print("Med: " + str(np.median(error_inst)))
    print("MSE: " + str(mse))
    print("RMSE: " + str(rmse))
    return error_inst, mse, rmse


def compute_error(model, 
                                    x_train_encode, y_train_encode,
                                    x_test_encode, y_test_encode
                                    , preprocessor):

    print("TRAINING ENCODE")
    pred_encode = model.predict(x_train_encode)
    error_inst, mse, rmse = calc_metrics(pred_encode, y_train_encode)
    
    print("TRAINING DECODE")
    prediction = model.predict(x_train_encode)
    x_decode, pred_decode = preprocessor.invert_transform(x_train_encode, prediction)
    x_decode, y_train_decode = preprocessor.invert_transform(x_train_encode, y_train_encode)
    error_inst, mse, rmse = calc_metrics(pred_decode, y_train_decode)

    print("TESTING ENCODE")
    prediction = model.predict(x_test_encode)
    error_inst, mse, rmse = calc_metrics(prediction, y_test_encode)
    
    print("TESTING DECODE")
    prediction = model.predict(x_test_encode)
    x_decode, pred_decode = preprocessor.invert_transform(x_test_encode, prediction)
    x_decode, y_test_decode = preprocessor.invert_transform(x_test_encode, y_test_encode)
    error_inst, mse, rmse = calc_metrics(pred_decode, y_test_decode)

    print("MAX DECODE PRED:" + str(pred_decode.max()))
    print("MAX DECODE TRUTH:"  + str(y_train_decode.max()))
    print("MIN DECODE PRED:"  + str(pred_decode.min()))
    print("MIN DECODE TRUTH:"  + str(y_train_decode.min()))
    return 

def compute_percent_median_error(predicted, truth):
        predicted = predicted.reshape(len(truth),3)
        truth = truth.reshape(len(truth),3)

        error = np.zeros((4,))
        cumulativeSum = 0.0
        zeros = np.zeros((4,))

        error[0] = np.median(np.abs((predicted[:,0] - truth[:,0])/ truth[:,0]))
        error[1] = np.median(np.abs((predicted[:,1] - truth[:,1])/ truth[:,1]))
        error[2] = np.median(np.abs((predicted[:,2] - truth[:,2])/ truth[:,2]))
        error[3] =  np.median(np.abs(np.linalg.norm(predicted - truth,axis=1)/ np.linalg.norm(truth,axis=1)))
        error *= 100

        print("\n\n\n")
        print("Median Total Error: " + str(error[3]) + "\n")
        print("Component Error")
        print(error[0:3])
        return 
def generate_experiment_dir(trajectory, preprocessor):
    experiment_dir = trajectory.trajectory_name + preprocessor.__class__.__name__
    experiment_dir =  experiment_dir.replace(', ', '_')
    experiment_dir =  experiment_dir.replace('[', '_')
    experiment_dir =  experiment_dir.replace(']', '_')
    return experiment_dir