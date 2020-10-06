from matplotlib import image
import numpy as np
data = image.imread('./Files/ShapeModels/peppers.jpg')

y = np.linspace(0, len(data)-1, len(data))
x = np.linspace(0, len(data[0])-1, len(data[0]))
z = data

input_vec = [] 
output_vec = [] 
for i in range(len(x)):
    for j in range(len(y)):
        input_vec.append([x[i], y[j]])
        output_vec.append([z[i][j]])

input_vec = np.array(input_vec)
output_vec = np.array(output_vec)

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

val_range = (0,1)
x_scaler = MinMaxScaler(feature_range=val_range)
y_scaler = MinMaxScaler(feature_range=val_range) 
#z_scaler = StandardScaler() 
z_scaler = MinMaxScaler() 


xy_train, xy_test, z_train, z_test = train_test_split(input_vec, output_vec, test_size=0.3, random_state=43)

x_train_encoded = x_scaler.fit_transform(xy_train[:,0].reshape(len(xy_train), 1))[:,0]
y_train_encoded= y_scaler.fit_transform(xy_train[:,1].reshape(len(xy_train), 1))[:,0]
x_test_encoded = x_scaler.transform(xy_test[:,0].reshape(len(xy_test), 1))[:,0]
y_test_encoded = y_scaler.transform(xy_test[:,1].reshape(len(xy_test), 1))[:,0]
z_train_encoded = z_scaler.fit_transform(z_train.reshape(len(z_train), 1))[:,0]
z_test_encoded = z_scaler.fit_transform(z_test.reshape(len(z_test), 1))[:,0]

input_train = np.stack((x_train_encoded, y_train_encoded), axis=1)
output_train = np.array([z_train_encoded]).transpose()

input_test = np.stack((x_test_encoded, y_test_encoded),axis=1)
output_test = np.array([z_test_encoded]).transpose()


# %%
from GravNN.GravityModels.NNSupport.NN_hyperparam import NN_hyperparam
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2

params = {}
params['epochs'] = 50
params['batch_size'] = 1
params['optimizer'] = Adadelta
params['kernel_initializer'] = GlorotUniform
params['kernel_regularizer'] = None
params['first_unit'] = 8
params['first_neuron'] =  8 #128
params['hidden_layers'] = 10
params['dropout'] = 0.1
params['lr'] = 0.1
params['shapes'] = 'brick'
params['activation'] = 'relu'
params['losses'] = 'mean_squared_error'# 'mean_squared_error'

save_location = None
hist, model = NN_hyperparam(input_train, output_train, input_test, output_test, params, verbose=1, save_location=save_location, validation_split=0.2)


# %%
import matplotlib.pyplot as plt
input_vec_encode = input_vec
input_vec_encode[:,0] = x_scaler.transform(input_vec_encode[:,0].reshape(len(input_vec_encode), 1))[:,0]
input_vec_encode[:,1] = y_scaler.transform(input_vec_encode[:,1].reshape(len(input_vec_encode), 1))[:,0]

pred = model.predict(input_vec_encode)
pred_decode = z_scaler.inverse_transform(pred.reshape(len(pred), 1))[:,0]
pred_decode = pred_decode.reshape(len(x),len(y))

print("True Max: " + str(output_vec.max()))
print("True Min: " + str(output_vec.min()))

print("Encode Max: " + str(pred.max()))
print("Encode Min: " + str(pred.min()))
print("Decode Max: " + str(pred_decode.max()))
print("Decode Min: " + str(pred_decode.min()))

plt.imshow(pred_decode,  cmap='gray')
plt.show()