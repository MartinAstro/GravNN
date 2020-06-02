import os
from CelestialBodies.CelestialBodyBase import CelestialBodyBase

class Earth(CelestialBodyBase):
    def __init__(self):
        self.body_name = "earth"
        self.grav_info.mu =  0.3986004415E+15  # meters^3/s^2
        self.grav_info.SH.sh_file = os.path.dirname(os.path.realpath(__file__))  + "/../Files/GravityModels/GGM03S.txt"

        self.geometry.radius = 6378136.6 # meters


class Moon(CelestialBodyBase):
    def __init__(self):
        self.body_name = "moon"
        self.grav_info.mu =  4.902799E12  # meters^3/s^2
        self.grav_info.SH.sh_file = os.path.dirname(os.path.realpath(__file__)) + "/../Files/GravityModels/GRAIL_1200a_sha.txt"

        self.geometry.radius = 1738100.0 # meters
