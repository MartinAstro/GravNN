import os
import numpy as np


class Bennu:
    def __init__(self):
        self.body_name = "bennu"
        self.density = 1260.0  # kg/m^3  https://github.com/bbercovici/SBGAT/blob/master/SbgatCore/include/SbgatCore/Constants.hpp
        self.radius = 282.37  # meters
        self.min_radius = 240.00
        G = 6.67430 * 10 ** -11
        self.mu = G * 7.329 * 10 ** 10  # self.density*(4./3.)*np.pi*self.radius**3*G
        self.obj_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Bennu-Radar.obj"
        )
        self.obj_hf_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/g_06290mm_spc_obj_0000n00000_v008.obj"
        )
        self.obj_vhf_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Bennu_v20_200k.stl"
        )

        self.sh_obj_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/GravityModels/Bennu-Radar_39sh.json"
        )
        self.sh_obj_hf_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/GravityModels/g_06290mm_spc_obj_0000n00000_v008_39sh.json"
        )
        self.obj_200k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Bennu_v20_200k.obj"
        )
        self.stl_200k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Bennu_v20_200k.stl"
        )

class Eros:
    def __init__(self):
        self.body_name = "eros"
        self.density = 2670.0  # kg/m^3 https://ssd.jpl.nasa.gov/sbdb.cgi#top
        self.physical_radius = (
            np.linalg.norm(np.array([34.4, 11.2, 11.2]) * 1e3) / 2
        )  # 34400.0/2.0# 16840.0 is *mean* diameter # meters (diameter / 2)
        self.radius = 16000.0  # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.998.2986&rep=rep1&type=pdf

        G = 6.67430 * 10 ** -11
        self.mu = G * 6.687 * 10 ** 15
        self.model_3k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_Blender_3k_poly.stl"
        )
        self.model_6k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_Blender_6k_poly.stl"
        )
        self.model_12k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_Blender_12k_poly.stl"
        )

        # mac only
        self.model_17k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_Gaskell_17k_poly.stl"
        )
        self.model_potatok = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_Blender_potato_k_poly.obj"
        )

        self.model_25k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_Blender_25k_poly.stl"
        )
        self.model_50k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_Gaskell_50k_poly.obj"
        )
        self.model_100k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_Blender_98k_poly.stl"
        )

        self.model_data = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_data_2.obj"
        )  # https://sbn.psi.edu/pds/resource/nearbrowse.html
        self.model_7790 = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Eros_Near_Shape_7790_facets2.obj"
        )  # https://sbn.psi.edu/pds/resource/nearbrowse.html

        self.sh_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/GravityModels/Regressed/Eros/true.csv"
        )
        
        self.obj_200k = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/eros200700.obj"
        )

class Toutatis:
    def __init__(self):
        self.body_name = "toutatis"

        G = 6.67430 * 10 ** -11

        # https://3d-asteroids.space/asteroids/4179-Toutatis
        self.model_lf = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Toutatis_Radar_based_Blender_lo_res.obj"
        )
        self.model_hf = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Toutatis_Radar_based_Blender_hi_res.obj"
        )

        # Scheeres Paper
        # volume of 7.670 km^3
        # density = 2.5/1000.0*100**3 # kg/m^3 -- 2.5 g/cm^3 (according to Dynamics of Orbits close to Toutatis -- Scheeres)
        self.radius = 1.223 * 1000
        self.density = 2.5 / 1000.0 * 100 ** 3
        self.mass = 1.917 * 10 ** 13

        self.mu = G * self.mass

        # from wiki
        # self.density = 2.1/1000.0*100**3 # kg/m^3 -- 2.1 g/cm^3 (Wiki)
        # self.radius = 5.4*1E3/2 # mean diameter from https://ssd.jpl.nasa.gov/sbdb.cgi?sstr=4179
