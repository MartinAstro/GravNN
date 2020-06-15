import os
class Earth():
    def __init__(self, gravityModel=None):
        self.body_name = "earth"
        self.mu =  0.3986004415E+15  # meters^3/s^2
        self.radius = 6378136.6 # meters
        self.gravModel = gravityModel
        self.sh_file = os.path.dirname(os.path.realpath(__file__))  + "/../Files/GravityModels/GGM03S.txt"    
        self.sh_hf_file = "/Users/johnmartin/Documents/GraduateSchool/Research/SH_GPU/GravityCompute/EGM2008_to2190_TideFree_E.txt"   



class Moon():
    def __init__(self, gravityModel=None):
        self.body_name = "moon"
        self.radius = 1738100.0 # meters
        self.mu =  4.902799E12  # meters^3/s^2
        self.gravModel = gravityModel
        self.sh_file = os.path.dirname(os.path.realpath(__file__)) + "/../Files/GravityModels/GRAIL_1200a_sha.txt"

