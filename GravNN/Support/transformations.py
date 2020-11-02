import numpy as np
import glob
import pickle
from numba import njit, prange

@njit(parallel=True, cache=True)
def sphere2cart(data):
    """Convert spherical coordinates into cartesian coordinates. Spherical coordinates should be in degrees.

    Args:
        data: [[3xM] with r, θ (0,360), Φ (0,180)]

    Returns:
        [np.array]: [cartesian output dimension [3xM]]
    """
    try:
        data[:,0]
    except:
        data = np.array(data)
    r = data[:,0] #[0, inf]
    theta = data[:,1] * np.pi / 180.0 # [0, 360]
    phi = data[:,2]* np.pi / 180.0 # [0, 180]

    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)

    return np.transpose(np.array([x,y,z]))

@njit(parallel=True,cache=True)
def cart2sph(carts):
    """Converts cartesian coordinates into spherical coordinates. Spherical coordinates should be in degrees.

    Args:
        carts: [[3xM] formatted as x, y, z]

    Returns:
        [np.array]: [[3xM] with r, θ (0,360), Φ (0,180)]
    """
    spheres = np.zeros(carts.shape)#spheres = []
    for i in prange(0,len(carts)):
        X, Y, Z = carts[i]

        r = np.sqrt(X**2 + Y**2 + Z**2)  # r
        theta = np.arctan2(Y, X) * 180.0 / np.pi + 180.0#  [0, 360]
        phi = np.arctan2(np.sqrt(X**2 + Y**2),Z) * 180.0 / np.pi   # [0,180]

        spheres[i] = [r, theta, phi]
    return spheres

def check_fix_radial_precision_errors(position):
    """Check the radial component of the vector to see if the values are all within machine precision. If so, round them to the same precision and value to ensure NN processing is not erroneous

    Args:
        position (np.array): positions in spherical coordinates

    Returns:
        np.array: positions in spherical coordinates
    """
    position = np.array(position)
    # If there is no diversity in the radial component, round them to be the same value. This is necessary for preprocessing. 
    if abs(position[:,0].max() - position[:,0].min()) < 10E-8:
        position[:,0] = np.round(position[:,0], 6)
    return position

@njit(parallel=True, cache=True)
def project_acceleration(positions, accelerations):
    """Converts the acceleration measurements from cartesian coordinates to spherical coordinates. 

    Args:
        positions (np.array): position in spherical coordinates
        accelerations (np.array): acceleration in cartesian coordinates

    Returns:
        np.array: acceleration in spherical coordinates
    """
    project_acc = np.zeros(accelerations.shape)
    for i in prange(0, len(positions)):
        theta = positions[i,1] * np.pi/180.0 - np.pi
        phi = positions[i,2] * np.pi/180.0 
        r_hat = np.array([np.sin(phi)*np.cos(theta),
                        np.sin(phi)*np.sin(theta),
                        np.cos(phi)])
        theta_hat = np.array([np.cos(phi)*np.cos(theta),
                                np.cos(phi)*np.sin(theta), 
                                -np.sin(phi)])
        phi_hat = np.array([-np.sin(theta), 
                            np.cos(theta),
                            0.0])
        project_acc[i,:] = np.array([np.dot(accelerations[i], r_hat),
                                    np.dot(accelerations[i], theta_hat),
                                    np.dot(accelerations[i], phi_hat)])
    return project_acc

@njit(parallel=True, cache=True)
def invert_projection(positions, accelerations):
    """Converts the acceleration measurements from spherical coordinates to cartesian coordinates

    Args:
        positions (np.array): position in spherical coordinates
        accelerations (np.array): acceleration in spherical coordinates

    Returns:
        np.array: acceleration in cartesian coordinates
    """
    invert_acc = np.zeros(np.shape(accelerations))
    for i in prange(0, len(positions)):
        theta = positions[i,1] * np.pi/180.0 - np.pi
        phi = positions[i,2] * np.pi/180.0 
        r_hat = [np.sin(phi)*np.cos(theta),
                        np.sin(phi)*np.sin(theta),
                        np.cos(phi)]
        theta_hat = [np.cos(phi)*np.cos(theta),
                                np.cos(phi)*np.sin(theta), 
                                -np.sin(phi)]
        phi_hat = [-np.sin(theta), 
                            np.cos(theta),
                            0]

        BN = np.array([r_hat,
                        theta_hat,
                        phi_hat])
                                    
        NB = np.transpose(BN)
        invert_acc[i,:] = np.dot(NB, accelerations[i])
    return invert_acc