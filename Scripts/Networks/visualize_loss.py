''' Visualize the magnitudes of each loss component against one another'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

df = pd.read_csv('history.csv', sep=",")
loss_components = []
for i in range(len(df['loss_components'])):
    to_ints = ast.literal_eval(df['loss_components'][i])
    loss_components.append(to_ints)

percent_components = []
for i in range(len(df['loss_components'])):
    to_ints = ast.literal_eval(df['loss_components'][i])
    loss_components.append(to_ints)

general_x = [j[0] for j in loss_components]
general_y = [j[1] for j in loss_components]
general_z = [j[2] for j in loss_components]
laplace = [j[3] for j in loss_components]
curl_x = [j[4] for j in loss_components]
curl_y = [j[5] for j in loss_components]
curl_z = [j[6] for j in loss_components ]

general_x_percent = [j[0] for j in percent_components]
general_y_percent = [j[1] for j in percent_components]
general_z_percent = [j[2] for j in percent_components]
laplace_percent = [j[3] for j in percent_components]
curl_x_percent = [j[4] for j in percent_components]
curl_y_percent = [j[5] for j in percent_components]
curl_z_percent = [j[6] for j in percent_components]

epochs = range(1,100)

plt.plot(epochs, general_x, 'b', label='General Loss X')
plt.plot(epochs, general_y, 'g', label='General Loss Y')
plt.plot(epochs, general_z, 'r', label='General Loss Z')
plt.plot(epochs, laplace, 'c', label='Laplace Loss')
plt.plot(epochs, curl_x, 'm', label='Curl Loss X')
plt.plot(epochs, curl_y, 'y', label='Curl Loss Y')
plt.plot(epochs, curl_z, 'k', label='Curl Loss Z')

plt.plot(epochs, general_x_percent, 'b', label='General Loss X Percent')
plt.plot(epochs, general_y_percent, 'g', label='General Loss Y Percent')
plt.plot(epochs, general_z_percent, 'r', label='General Loss Z Percent')
plt.plot(epochs, laplace_percent, 'c', label='Laplace Loss Percent')
plt.plot(epochs, curl_x_percent, 'm', label='Curl Loss X Percent')
plt.plot(epochs, curl_y_percent, 'y', label='Curl Loss Y Percent')
plt.plot(epochs, curl_z_percent, 'k', label='Curl Loss Z Percent')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()