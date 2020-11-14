import os
class Bennu():
    def __init__(self, gravityModel=None):
        self.body_name = "bennu"
        self.density =  1260.0  # kg/m^3  https://github.com/bbercovici/SBGAT/blob/master/SbgatCore/include/SbgatCore/Constants.hpp 
        self.radius = 282.37 # meters
        self.obj_file = os.path.dirname(os.path.realpath(__file__))  + "/../Files/ShapeModels/Bennu-Radar.obj"    
        self.obj_hf_file = os.path.dirname(os.path.realpath(__file__)) +"/../Files/ShapeModels/g_06290mm_spc_obj_0000n00000_v008.obj" 
        self.sh_obj_file = os.path.dirname(os.path.realpath(__file__)) + "/../Files/GravityModels/Bennu-Radar_39sh.json" 
        self.sh_obj_hf_file = os.path.dirname(os.path.realpath(__file__)) + "/../Files/GravityModels/g_06290mm_spc_obj_0000n00000_v008_39sh.json" 

