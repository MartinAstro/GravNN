import numpy as np

from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.ExponentialDist import ExponentialDist
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Support.transformations import cart2sph



def main():
    planet = Earth()
    trajectory = RandomDist(planet, [planet.radius, planet.radius+420000.0], 1000000)

    positions_sph = cart2sph(trajectory.positions)

    sort_theta = np.sort(positions_sph[:,1])
    sort_phi = np.sort(positions_sph[:,2])

    sort_theta_prime = np.roll(sort_theta, 1)
    sort_phi_prime = np.roll(sort_phi,1)

    delta_theta = sort_theta - sort_theta_prime
    delta_phi = sort_phi - sort_phi_prime

    delta_theta[0] += 360
    delta_phi[0] += 180

    max_theta = np.max(delta_theta)
    max_phi = np.max(delta_phi)

    print(180.0/np.max([max_theta, max_phi]))

    # Option 2, find nearest theta, and measure distance too it
    positions_by_theta = positions_sph[positions_sph[:,1].argsort()]
    diff = positions_by_theta - np.roll(positions_by_theta, 1, axis=0)
    diff[0,1] += 360
    distances = np.linalg.norm(diff[:,1:])

    print(np.max(distances))
    print(180.0/ np.max(distances))

    # Option 3, find nearest point in angular distance

    max_dist_global = 0
    for position in positions_sph:
        diff = positions_sph - position

        idx = np.where(diff[:,1] > 180)
        diff[idx,1] -= 360

        diff_mag = np.linalg.norm(diff[:,1:],axis=1)
        max_dist = np.min(diff_mag[diff_mag!=0])
        if max_dist_global < max_dist:
            max_dist_global = max_dist
    
    print(max_dist_global)
    print(180.0/max_dist_global)
    



if __name__=='__main__':
    main()