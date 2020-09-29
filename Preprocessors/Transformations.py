from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


#%% Pure
def minmax_all(grid):
    r_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,0]]))
    theta_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,1]]))
    phi_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,2]]))

    acc_r_scalar = MinMaxScaler().fit_transform(np.transpose([grid.acceleration[:,0]]))
    acc_theta_scalar = MinMaxScaler().fit_transform(np.transpose([grid.acceleration[:,1]]))
    acc_phi_scalar = MinMaxScaler().fit_transform(np.transpose([grid.acceleration[:,2]]))

    data = np.transpose(np.vstack((r_scaler[:,0], theta_scaler[:,0], phi_scaler[:,0], acc_r_scalar[:,0], acc_theta_scalar[:,0], acc_phi_scalar[:,0])))
    return data


def standard_all_grid(grid):
    r_scaler = StandardScaler().fit_transform(np.transpose([grid.positions[:,0]]))
    theta_scaler = StandardScaler().fit_transform(np.transpose([grid.positions[:,1]]))
    phi_scaler = StandardScaler().fit_transform(np.transpose([grid.positions[:,2]]))

    acc_r_scalar = StandardScaler().fit_transform(np.transpose([grid.acceleration[:,0]]))
    acc_theta_scalar = StandardScaler().fit_transform(np.transpose([grid.acceleration[:,1]]))
    acc_phi_scalar = StandardScaler().fit_transform(np.transpose([grid.acceleration[:,2]]))

    data = np.transpose(np.vstack((r_scaler[:,0], theta_scaler[:,0], phi_scaler[:,0], acc_r_scalar[:,0], acc_theta_scalar[:,0], acc_phi_scalar[:,0])))
    return data


def standard_all(data):
    for i in range(len(data[0])):
        data[:,i] = StandardScaler().fit_transform(np.transpose([data[:,i]]))[:,0]
    return data

#%% Hybrid
def minmax_pos_standard_acc(grid):
    """Each position component scaled by Min Max
    Each acceleration component scaled by Standard
    """
    r_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,0]]))
    theta_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,1]]))
    phi_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,2]]))

    acc_r_scalar = StandardScaler().fit_transform(np.transpose([grid.acceleration[:,0]]))
    acc_theta_scalar = StandardScaler().fit_transform(np.transpose([grid.acceleration[:,1]]))
    acc_phi_scalar = StandardScaler().fit_transform(np.transpose([grid.acceleration[:,2]]))

    data = np.transpose(np.vstack((r_scaler[:,0], theta_scaler[:,0], phi_scaler[:,0], acc_r_scalar[:,0], acc_theta_scalar[:,0], acc_phi_scalar[:,0])))
    return data

def minmax_pos_standard_acc_mag(grid):
    """Each position component scaled by Min Max
    All acceleration component scaled by Standard of the acceleration magnitude
    """
    r_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,0]]))
    theta_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,1]]))
    phi_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,2]]))

    acc_norm = np.linalg.norm(grid.acceleration, axis=1)
    acc_norm_scaler = StandardScaler().fit(np.transpose([acc_norm]))

    acc_r_scalar = acc_norm_scaler.transform(np.transpose([grid.acceleration[:,0]]))
    acc_theta_scalar = acc_norm_scaler.transform(np.transpose([grid.acceleration[:,1]]))
    acc_phi_scalar = acc_norm_scaler.transform(np.transpose([grid.acceleration[:,2]]))

    data = np.transpose(np.vstack((r_scaler[:,0], theta_scaler[:,0], phi_scaler[:,0], acc_r_scalar[:,0], acc_theta_scalar[:,0], acc_phi_scalar[:,0])))
    return data

def minmax_pos_standard_acc_components(data):
    """Each position component scaled by Min Max
    Each acceleration component scaled by Standard
    """
    for i in range(0, 3):
        data[:,i] = MinMaxScaler().fit_transform(np.transpose([data[:,i]]))[:,0]

    for i in range(3, len(data[0])):
        data[:,i] = StandardScaler().fit_transform(np.transpose([data[:,i]]))[:,0]
    
    return data


def minmax_pos_standard_acc_latlon(grid):
    """Each position component scaled by Min Max
    Each acceleration component scaled by Standard
    """
    r_scaler = MinMaxScaler().fit_transform(np.transpose([grid.positions[:,0]]))
    LatLongScaler = MinMaxScaler()
    theta_scaler = LatLongScaler.fit_transform(np.transpose([grid.positions[:,1]]))
    phi_scaler = LatLongScaler.transform(np.transpose([grid.positions[:,2]]))

    acc_r_scalar = StandardScaler().fit_transform(np.transpose([grid.acceleration[:,0]]))
    acc_theta_scalar = StandardScaler().fit_transform(np.transpose([grid.acceleration[:,1]]))
    acc_phi_scalar = StandardScaler().fit_transform(np.transpose([grid.acceleration[:,2]]))

    data = np.transpose(np.vstack((r_scaler[:,0], theta_scaler[:,0], phi_scaler[:,0], acc_r_scalar[:,0], acc_theta_scalar[:,0], acc_phi_scalar[:,0])))
    return data

def minmax_pos_minmax_standard_acc_latlon(data):
    """Each position component scaled by Min Max
    Each acceleration component scaled by Standard
    """
    r_scaler = MinMaxScaler().fit_transform(np.transpose([data[:,0]]))
    LatLongScaler = MinMaxScaler()
    theta_scaler = LatLongScaler.fit_transform(np.transpose([data[:,1]]))
    phi_scaler = LatLongScaler.transform(np.transpose([data[:,2]]))

    acc_r_scalar = MinMaxScaler().fit_transform(np.transpose([data[:,3]]))
    acc_theta_scalar = MinMaxScaler().fit_transform(np.transpose([data[:,4]]))
    acc_phi_scalar = MinMaxScaler().fit_transform(np.transpose([data[:,5]]))

    acc_r_scalar = StandardScaler().fit_transform(acc_r_scalar)
    acc_theta_scalar = StandardScaler().fit_transform(acc_theta_scalar)
    acc_phi_scalar = StandardScaler().fit_transform(acc_phi_scalar)

    data = np.transpose(np.vstack((r_scaler[:,0], theta_scaler[:,0], phi_scaler[:,0], acc_r_scalar[:,0], acc_theta_scalar[:,0], acc_phi_scalar[:,0])))
    return data