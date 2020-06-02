import numpy as np
import glob
import pickle

def sphere2cart(data):
    r = data[:,0] #[0, inf]
    theta = data[:,1] * np.pi / 180.0 # [0, 360]
    phi = data[:,2]* np.pi / 180.0 # [0, 180]

    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)

    return np.transpose(np.array([x,y,z]))


def cart2sph(carts):
    spheres = []
    for cartFull in carts:
        if len(cartFull) == 4:
            X,Y,Z = cartFull[1:4]
        else:
            X, Y, Z = cartFull

        r = np.sqrt(X**2 + Y**2 + Z**2)  # r
        theta = np.arctan2(Y, X) * 180.0 / np.pi + 180.0# theta [0, 360]
        phi = np.arctan2(np.sqrt(X**2 + Y**2),Z) * 180.0 / np.pi   # Should be [0,180]

        if len(cartFull) == 4:
            spheres.append([cartFull[0], r, theta, phi])
        else:
            spheres.append([r, theta, phi])
    return np.array(spheres)