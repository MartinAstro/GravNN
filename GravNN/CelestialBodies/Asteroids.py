import os
import numpy as np
class Bennu():
    def __init__(self, gravityModel=None):
        self.body_name = "bennu"
        self.density =  1260.0  # kg/m^3  https://github.com/bbercovici/SBGAT/blob/master/SbgatCore/include/SbgatCore/Constants.hpp 
        self.radius = 282.37 # meters

        G = 6.67430*10**-11
        self.mu = G*7.329*10**10 # self.density*(4./3.)*np.pi*self.radius**3*G
        self.obj_file = os.path.dirname(os.path.realpath(__file__))  + "/../Files/ShapeModels/Bennu-Radar.obj"    
        self.obj_hf_file = os.path.dirname(os.path.realpath(__file__)) +"/../Files/ShapeModels/g_06290mm_spc_obj_0000n00000_v008.obj" 
        self.obj_vhf_file = os.path.dirname(os.path.realpath(__file__)) +"/../Files/ShapeModels/Bennu_v20_200k.stl" 

        self.sh_obj_file = os.path.dirname(os.path.realpath(__file__)) + "/../Files/GravityModels/Bennu-Radar_39sh.json" 
        self.sh_obj_hf_file = os.path.dirname(os.path.realpath(__file__)) + "/../Files/GravityModels/g_06290mm_spc_obj_0000n00000_v008_39sh.json" 


class Eros():
    def __init__(self, gravityModel=None):
        self.body_name = "eros"
        self.density =  2670.0  # kg/m^3 https://ssd.jpl.nasa.gov/sbdb.cgi#top
        self.radius = np.linalg.norm(np.array([34.4, 11.2, 11.2])*1E3)/2  #34400.0/2.0# 16840.0 is *mean* diameter # meters (diameter / 2)

        G = 6.67430*10**-11
        #self.mu = G*7.329*10**10 # self.density*(4./3.)*np.pi*self.radius**3*G
        self.model_3k = os.path.dirname(os.path.realpath(__file__))  + "/../Files/ShapeModels/Eros_Blender_3k_poly.stl"    
        self.model_6k = os.path.dirname(os.path.realpath(__file__))  + "/../Files/ShapeModels/Eros_Blender_6k_poly.stl"    
        self.model_12k = os.path.dirname(os.path.realpath(__file__))  + "/../Files/ShapeModels/Eros_Blender_12k_poly.stl"    

        # mac only
        self.model_17k = os.path.dirname(os.path.realpath(__file__))  + "/../Files/ShapeModels/Eros_Gaskell_17k_poly.stl"   
        self.model_potatok = os.path.dirname(os.path.realpath(__file__))  + "/../Files/ShapeModels/Eros_Blender_potato_k_poly.obj"   

        self.model_25k = os.path.dirname(os.path.realpath(__file__))  + "/../Files/ShapeModels/Eros_Blender_25k_poly.stl"    
        self.model_50k = os.path.dirname(os.path.realpath(__file__))  + "/../Files/ShapeModels/Eros_Gaskell_50k_poly.obj"  
        self.model_100k = os.path.dirname(os.path.realpath(__file__)) +"/../Files/ShapeModels/Eros_Blender_98k_poly.stl"   
        #self.obj_hf_file = os.path.dirname(os.path.realpath(__file__)) +"/../Files/ShapeModels/Eros_Gaskell_200k_poly.obj" 
        # self.obj_hf_file = os.path.dirname(os.path.realpath(__file__)) +"/../Files/ShapeModels/Eros_Blender_98k_poly.obj" 
        # self.stl_hf_file = os.path.dirname(os.path.realpath(__file__)) +"/../Files/ShapeModels/Eros_Blender_98k_poly.stl" 

        # self.sh_obj_file = os.path.dirname(os.path.realpath(__file__)) + "/../Files/GravityModels/Bennu-Radar_39sh.json" 
        # self.sh_obj_hf_file = os.path.dirname(os.path.realpath(__file__)) + "/../Files/GravityModels/g_06290mm_spc_obj_0000n00000_v008_39sh.json" 

