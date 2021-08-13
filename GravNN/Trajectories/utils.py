from GravNN.Trajectories import EphemerisDist
from GravNN.CelestialBodies.Asteroids import Eros


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