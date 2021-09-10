from GravNN.Trajectories import EphemerisDist, RandomAsteroidDist
from GravNN.CelestialBodies.Asteroids import Eros
import numpy as np

def generate_near_orbit_trajectories(sampling_inteval):
    orbits = [
        "Feb 24, 2000",
        "Mar 03, 2000",
        "Apr 02, 2000",
        "Apr 11, 2000",
        "Apr 22, 2000",
        "Apr 30, 2000",
        "July 07, 2000",
        "July 14, 2000",
        "July 24, 2000",
        "July 31, 2000",
        "Aug 08, 2000",
        "Aug 26, 2000",
        "Sep 05, 2000",
        "Oct 13, 2000",
        "Oct 20, 2000",
        "Oct 25, 2000",
        "Oct 26, 2000",
        "Nov 03, 2000",
        "Dec 07, 2000",
        "Dec 13, 2000",
        "Jan 24, 2001",
        "Jan 28, 2001",
        "Jan 29, 2001",
        "Feb 02, 2001",
        "Feb 06, 2001",
    ]

    trajectories = []
    # https://www.jhuapl.edu/Content/techdigest/pdf/V23-N01/23-01-Holdridge.pdf -- NEAR Orbits
    for i in range(0, len(orbits) - 1):
        trajectory = EphemerisDist(
            source="NEAR",
            target="EROS",
            frame="EROS_FIXED",
            start_time=orbits[i],
            end_time=orbits[i + 1],
            sampling_interval=sampling_inteval,
            celestial_body=Eros(),
        )
        trajectories.append(trajectory)
    return trajectories


def generate_near_hopper_trajectories(sampling_inteval):
    orbits = [
        "Feb 24, 2000",
        "Mar 03, 2000",
        "Apr 02, 2000",
        "Apr 11, 2000",
        "Apr 22, 2000",
        "Apr 30, 2000",
        "July 07, 2000",
        "July 14, 2000",
        "July 24, 2000",
        "July 31, 2000",
        "Aug 08, 2000",
        "Aug 26, 2000",
        "Sep 05, 2000",
        "Oct 13, 2000",
        "Oct 20, 2000",
        "Oct 25, 2000",
        "Oct 26, 2000",
        "Nov 03, 2000",
        "Dec 07, 2000",
        "Dec 13, 2000",
        "Jan 24, 2001",
        "Jan 28, 2001",
        "Jan 29, 2001",
        "Feb 02, 2001",
        "Feb 06, 2001",
    ]

    trajectories = []
    # https://www.jhuapl.edu/Content/techdigest/pdf/V23-N01/23-01-Holdridge.pdf -- NEAR Orbits
    for i in range(0, len(orbits) - 1):
        trajectory = EphemerisDist(
            source="NEAR",
            target="EROS",
            frame="EROS_FIXED",
            start_time=orbits[i],
            end_time=orbits[i + 1],
            sampling_interval=sampling_inteval,
            celestial_body=Eros(),
        )
        trajectories.append(trajectory)
    
    hopper_trajectories = []
    planet = Eros()
    for i in range(len(orbits)-1):
        trajectory = trajectories[i]
        total_time = trajectory.times[-1] - trajectory.times[0]
        samples = total_time / sampling_inteval # 1 image every 10 minutes
        samples *= 10 # 10 hoppers
        samples /= 2 # only half are visible in an image (assuming distribution is somewhat equally distributed around the asteroid)
        samples /= 50 # takes 50 images to produce a robust estimate of the arc (should be 1 percent of the original data)
        samples /= 10 # can only image the targets 10% of the time (otherwise using spacecraft for science, charging, or maintenance)
        trajectory = RandomAsteroidDist(planet, [0, 1.05*planet.radius],int(samples), model_file=planet.obj_200k)
        hopper_trajectories.append(trajectory)
    return hopper_trajectories

def single_near_trajectory():
    trajectories = generate_near_orbit_trajectories(60*10)
    trajectory = trajectories[0]
    for i in range(1,len(trajectories)):
        trajectory.positions = np.concatenate((trajectory.positions, trajectories[i].positions), axis=0)
        trajectory.accelerations = np.concatenate((trajectory.accelerations, trajectories[i].accelerations), axis=0)
        trajectory.times = np.concatenate((trajectory.times, trajectories[i].times), axis=0)

    return trajectory