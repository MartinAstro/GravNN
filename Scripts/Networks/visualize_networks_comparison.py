'''
Plot visualization of different model hyperparameters against their corresponding errors
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read to dataframe
column_names = ["Model", "Error", "Error_STD", "Time", "Layers", "Batch_Size", "Learning_Rate", "Activation"]
df = pd.read_csv('test_hyperparameters.txt', sep=",", names=column_names)

# Check for nulls and drop
df = df.dropna()

# Make graphs
print("Smallest Error: \n")
best_models = df.nsmallest(20, 'Error')
best_batch_size = best_models["Batch_Size"].mode()
best_learning_rate = best_models["Learning_Rate"].mode()
best_activation = best_models["Activation"].mode()
best_layers = best_models["Layers"].mode()
print(best_models["Layers"])
print(best_models["Layers"].nsmallest())

print("Best Models: \n")
print(best_models)
print("Batch Size: " + str(best_batch_size))
print("Learning Rate: " + str(best_learning_rate))
print("Activation: " + str(best_activation))
print("Layers: " + str(best_layers))

plt.figure(1)
plt.plot(df["Learning_Rate"].tolist(), df["Error"].tolist(), 'bo')
plt.xlabel("Learning Rate")
plt.ylabel("Error")
plt.title("Learning Rate vs. Error")
plt.show()

plt.figure(2)
plt.plot(df["Batch_Size"].tolist(), df["Error"].tolist(), 'bo')
plt.xlabel("Batch Size")
plt.ylabel("Error")
plt.title("Batch Size vs. Error")
plt.show()

plt.figure(3)
plt.plot(df["Layers"].tolist(), df["Error"].tolist(), 'bo')
plt.xlabel("Layers")
plt.ylabel("Error")
plt.title("Layers vs. Error")
plt.show()

plt.figure(4)
plt.plot(df["Activation"].tolist(), df["Error"].tolist(), 'bo')
plt.xlabel("Activation")
plt.ylabel("Error")
plt.title("Activation vs. Error")
plt.show()