import numpy as np
import matplotlib.pyplot as plt
def plot_metrics(history, save_location=None):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_location is not None:
        plt.savefig(plt.gcf(), save_location + "loss.pdf")
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

def generate_experiment_dir(trajectory, preprocessor):
    experiment_dir = trajectory.trajectory_name + preprocessor.__class__.__name__
    experiment_dir =  experiment_dir.replace(', ', '_')
    experiment_dir =  experiment_dir.replace('[', '_')
    experiment_dir =  experiment_dir.replace(']', '_')
    return experiment_dir