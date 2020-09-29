import numpy as np
import umap

def project_12DOF(data):
    for i in range(len(data[0])):
        new_column = data[:,i]*data[:,i+1]
        data = np.append(data, new_column, axis=1)

def project_42DOF(data):    
    original_dim = len(data[0])
    for i in range(original_dim):
        for j in range(original_dim):
            new_column = np.reshape(data[:,i]*data[:,j], (len(data),1))
            data = np.append(data, new_column, axis=1)
    return data

def project_18DOF(data):    
    original_dim = len(data[0])
    data_squared = data*data
    data = np.append(data, np.roll(data, 3, axis=1)*data, axis=1)
    data = np.append(data, data_squared,axis=1)
    return data

def project_none(data):
    return data

def project_UMAP(data):
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.0)
    embedding = reducer.fit_transform(data)
    return embedding 