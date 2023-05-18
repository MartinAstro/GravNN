import numpy as np

from GravNN.Support.ProgressBar import ProgressBar


def batch_function(fcn, output_shape, input_data, batch_size, pbar=True):
    num_samples = input_data.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    output_data = np.zeros(output_shape)
    pbar = ProgressBar(num_batches, enable=pbar)
    for i in range(num_batches):
        start = i * batch_size
        end = np.min([(i + 1) * batch_size, num_samples])
        output_data[start:end] = fcn(input_data[start:end])
        pbar.update(i)
    pbar.update(i + 1)
    pbar.close()
    return output_data


# Rewrite batch_function, wrapping the for loop with a progress bar.
# Make the progress bar an optional input keyword.
