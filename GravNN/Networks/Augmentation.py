import numpy as np
import copy
def periodic(x, a, u):
    condition = np.abs(x[:,1] - np.pi) < 2*np.pi/10.0 and np.abs(x[:,2] - np.pi/2) < np.pi/10.0
    x_copy = copy.deepcopy(x[condition])
    a_copy = copy.deepcopy(a[condition])
    u_copy = copy.deepcopy(u[condition])

    for i in range(0,len(x_copy)):
        x_row = x_copy[i]

        value = -2*np.pi if x_row[1] >= np.pi else 2*np.pi
        x_row[1] = x_row[1] + value
        
        value = -np.pi if x_row[2] >= np.pi/2 else np.pi
        x_row[2] = x_row[2] + value

        x_copy[i] = x_row
    
    x_aug = np.concatenate([x,x_copy], axis=0)
    a_aug = np.concatenate([a,a_copy], axis=0)
    u_aug = np.concatenate([u,u_copy], axis=0)

    return x_aug, a_aug, u_aug
