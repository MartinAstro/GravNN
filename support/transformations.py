import numpy as np
import glob
import pickle

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


def cart2sph(carts):
    """Converts cartesian coordinates into spherical coordinates. Spherical coordinates should be in degrees.

    Args:
        carts: [[3xM] formatted as x, y, z]

    Returns:
        [np.array]: [[3xM] with r, θ (0,360), Φ (0,180)]
    """
    spheres = []
    for cartFull in carts:
        X, Y, Z = cartFull

        r = np.sqrt(X**2 + Y**2 + Z**2)  # r
        theta = np.arctan2(Y, X) * 180.0 / np.pi + 180.0#  [0, 360]
        phi = np.arctan2(np.sqrt(X**2 + Y**2),Z) * 180.0 / np.pi   # [0,180]

        spheres.append([r, theta, phi])
    return np.array(spheres)

def project_acceleration(positions, accelerations):
    """Converts the acceleration measurements from cartesian coordinates to spherical coordinates. 

    Args:
        positions (np.array): position in spherical coordinates
        accelerations (np.array): acceleration in cartesian coordinates

    Returns:
        np.array: acceleration in spherical coordinates
    """
    project_acc = np.zeros(np.shape(accelerations))
    for i in range(len(positions)):
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
        project_acc[i,:] = np.array([np.dot(accelerations[i], r_hat),
                                                        np.dot(accelerations[i], theta_hat),
                                                        np.dot(accelerations[i], phi_hat)])
    return project_acc

def invert_projection(positions, accelerations):
    """Converts the acceleration measurements from spherical coordinates to cartesian coordinates

    Args:
        positions (np.array): position in spherical coordinates
        accelerations (np.array): acceleration in spherical coordinates

    Returns:
        np.array: acceleration in cartesian coordinates
    """
    invert_acc = np.zeros(np.shape(accelerations))
    for i in range(len(positions)):
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