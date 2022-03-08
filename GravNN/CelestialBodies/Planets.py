import os
import re
import pooch
import GravNN
import numpy as np
from zipfile import ZipFile
class Earth:
    def __init__(self):
        self.body_name = "earth"
        self.mu = 0.3986004415e15  # meters^3/s^2
        self.radius = 6378136.6  # meters
        

        def unpack(fname, action, member):
            # manual unpack processor
            unzipped = fname + ".unzipped"
            # Don't unzip if file already exists and is not being downloaded
            if action in ("update", "download") or not os.path.exists(unzipped):
                with ZipFile(fname, "r") as zip_file:
                    # Extract the data file from within the archive
                    with zip_file.open(member) as data_file:
                        # Save it to our desired file name
                        with open(unzipped, "wb") as output:
                            output.write(data_file.read())
            # Return the path of the unzipped file
            return unzipped

        def format_EGM96_sh(fname, action, pooch_inst):
            unzipped = unpack(fname, action, "EGM96")
            new_name = fname.split("_raw")[0] + ".txt"
            if os.path.exists(new_name):
                return new_name

            with open(unzipped, 'rb') as f:
                data = f.readlines()
            os.remove(unzipped)

            with open(new_name, 'w') as f:
                f.write("%f,%f,%f,%d\n" % (self.radius, self.mu, 0.0, 360))
                f.write("0,0, 1.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00\n")
                f.write("1,0, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00\n")
                f.write("1,1, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00\n")
                f.writelines([ re.sub("\s+", ",", line.decode("utf-8").lstrip()) + "\n" for line in data])
            return new_name

        def format_EGM2008_sh(fname, action, pooch_inst):
            unzipped = unpack(fname, action, "EGM2008_to2190_TideFree")
            new_name = fname.split("_raw")[0] + ".txt"
            if os.path.exists(new_name):
                return new_name

            with open(unzipped, 'rb') as f:
                data = f.readlines()
            os.remove(unzipped)

            with open(new_name, 'w') as f:
                f.write("%f,%f,%f,%d\n" % (self.radius, self.mu, 0.0, 2190))
                f.write("0,0, 1.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00\n")
                f.write("1,0, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00\n")
                f.write("1,1, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00, 0.00000000000E+00\n")
                f.writelines([ re.sub("\s+", ",", line.decode("utf-8").replace("D", "E").lstrip()) + "\n" for line in data])
            return new_name

        self.EGM96 = pooch.retrieve(
            url='https://earth-info.nga.mil/php/download.php?file=egm-96spherical',
            known_hash="1f21ab8151c1b9fe25f483a4f6b78acdbf5306daf923725017b83d87a5f33472",
            fname="EGM96_raw.zip",
            path=os.path.dirname(GravNN.__file__) + "/Files/GravityModels/Earth/",
            processor=format_EGM96_sh
        )

        self.EGM2008 = pooch.retrieve(
            url='https://earth-info.nga.mil/php/download.php?file=egm-08spherical',
            known_hash='65a9072f337f156e8cbd76ffd773f536e6fb0de18697ea6726ecdb790fac0fbd',
            fname="EGM2008_raw.zip",
            path=os.path.dirname(GravNN.__file__) + "/Files/GravityModels/Earth/",
            processor=format_EGM2008_sh
        )

        self.shape_model = os.path.dirname(GravNN.__file__) + "/Files/ShapeModels/Misc/unit_sphere.obj"

        # Backwards compatability
        # self.sh_file = self.EGM96
        self.sh_file = self.EGM2008

class Moon:
    def __init__(self):
        self.body_name = "moon"
        self.radius = 1738100.0  # meters
        self.mu = 4.902799e12  # meters^3/s^2

        def format_sh(fname, action, pooch_inst):
            new_name = fname.split("_raw.txt")[0] + ".txt"
            if os.path.exists(new_name):
                return new_name

            with open(fname, 'r') as f:
                data = f.readlines()
            meta_data = np.array([float(x) for x in data[0].split(", ")])
            meta_data[0] *= 1000.0 # change radius units to meters
            meta_data[1] *= 1000.0**3 # change mu units to meters^3
            
            with open(new_name, 'w') as f:
                f.write(np.array2string(meta_data, separator=', ', max_line_width=2000)[1:-1] + "\n")
                f.write("    0,    0,  1.00000000000E+00,  0.00000000000E+00, 0.00000000000E+00,  0.00000000000E+00\n")
                f.writelines(data[1:])
            return new_name

        self.GRGM1200 = pooch.retrieve(
            url='https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200a_sha.tab',
            known_hash="fa04c3dce9376948ad243f3df74144e2602f12d183ea4d179604ed0a79da7ded",
            fname="GRGM_1200_raw.txt",
            path=os.path.dirname(GravNN.__file__) + "/Files/GravityModels/Moon/",
            processor=format_sh
        )
        
        self.sh_file = self.GRGM1200
        self.shape_model = os.path.dirname(GravNN.__file__) + "/Files/ShapeModels/Misc/unit_sphere.obj"
