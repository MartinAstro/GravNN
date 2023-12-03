import os

import numpy as np
import pooch

import GravNN


class Asteroid:
    def __init__(self):
        pass


class Bennu(Asteroid):
    def __init__(self):
        self.body_name = "bennu"
        self.density = 1260.0  # kg/m^3  https://github.com/bbercovici/SBGAT/blob/master/SbgatCore/include/SbgatCore/Constants.hpp
        self.radius = 282.37  # meters
        self.min_radius = 240.00
        G = 6.67430 * 10**-11
        self.mu = G * 7.329 * 10**10  # self.density*(4./3.)*np.pi*self.radius**3*G

        def remove_whitespace(fname, action, pooch_inst):
            "add 1 to the face indices in the obj file to work with trimesh"
            new_name = fname.split("_raw")[0] + ".obj"
            if os.path.exists(new_name):
                return new_name

            with open(fname, "r") as f:
                lines = f.readlines()

            for i in range(len(lines)):
                line = lines[i]
                lines[i] = line.strip() + "\n"

            with open(new_name, "w") as f:
                f.writelines(lines)

            return new_name

        self.obj_file = pooch.retrieve(
            url="http://www.asteroidmission.org/wp-content/uploads/2019/01/Bennu-Radar.obj",
            known_hash="0aa41b9ce4c366bb72120e872f5a604ce5766063e6744e76bd4a68ed0f1d4f75",
            fname="Bennu-Radar.obj",
            path=os.path.dirname(GravNN.__file__) + "/Files/ShapeModels/Bennu/",
        )

        self.obj_200k = pooch.retrieve(
            url="http://www.asteroidmission.org/wp-content/uploads/2019/03/Bennu_v20_200k.obj",
            known_hash="afbf196bf570d84804e9dd5935425d60eee2884ea58b02cd4d1ef45d215f67de",
            fname="Bennu_shape_200700k_raw.obj",
            path=os.path.dirname(GravNN.__file__) + "/Files/ShapeModels/Bennu/",
            processor=remove_whitespace,
        )

        # Spherical Harmonics
        def format_sh(fname, action, pooch_inst):
            new_name = fname.split("_raw.txt")[0] + ".txt"
            if os.path.exists(new_name):
                return new_name
            # Pull data from the .m file
            with open(fname, "r") as f:
                data = f.readlines()

            for i in range(len(data)):
                line = data[i]
                if "DEGREE" in line:
                    max_deg = int(line.split("=")[1].split(";")[0])
                if "NAMES" in line:
                    name_start_idx = i + 1
                if "};" in line:
                    if "BENNU" in line:
                        name_end_idx = i + 1
                    else:
                        name_end_idx = i
                if "VALS" in line:
                    values_start_idx = i + 1
                if "];" in line:
                    if "-" in line:
                        values_end_idx = i + 1
                    else:
                        values_end_idx = i
                    break

            names = data[name_start_idx:name_end_idx]
            values = data[values_start_idx:values_end_idx]

            # Gather keys and values
            name_list = []
            value_list = []
            for i in range(len(names)):
                name_entries = [
                    entry.replace(" ", "")
                    .replace("'", "")
                    .replace("BENNU_", "")
                    .replace("...", "")
                    for entry in names[i].split("' '")
                ]
                value_entries = values[i].split(" ")
                value_entries = np.unique(value_entries).tolist()
                value_entries.remove("")
                try:
                    value_entries.remove("...\n")  # [3:-1]
                except:
                    value_entries.remove("];\n")
                name_list = np.concatenate((name_list, name_entries))
                value_list = np.concatenate((value_list, value_entries))

            # Standardize formatting and populate data array
            data_matrix = np.zeros((max_deg * (max_deg + 1) - 2 * (2 + 1), 6))
            for k in range(len(name_list)):
                name = name_list[k]
                if "J" in name:
                    degree = name.split("J")[1]
                    if len(degree) > 1:
                        name_list[k] = "C" + degree + "00"
                    else:
                        name_list[k] = "C0" + degree + "00"
                if "GM" in name:
                    continue

                coef = name_list[k][0]
                i = int(name_list[k][1:3])
                j = int(name_list[k][3:5])
                value = float(value_list[k])

                if coef == "C":
                    data_matrix[(i - 2) * max_deg + j] = np.array(
                        [i, j, value, data_matrix[(i - 2) * max_deg + j, 3], 0.0, 0.0],
                    )
                else:
                    data_matrix[(i - 2) * max_deg + j] = np.array(
                        [i, j, data_matrix[(i - 2) * max_deg + j, 2], value, 0.0, 0.0],
                    )

            # Remove empty rows
            data_matrix = data_matrix[
                ~np.all(data_matrix == np.array([0, 0, 0, 0, 0, 0]), axis=1)
            ]

            # Write data to processed file
            with open(new_name, "w") as f:
                f.write("    %f    %f    %f    %d\n" % (self.radius, self.mu, 0.0, 16))
                f.write(
                    "    0\t0\t1.00000000000E+00\t0.00000000000E+00\t0.00000000000E+00\t0.00000000000E+00\n",
                )
                f.write(
                    "    1\t0\t0.00000000000E+00\t0.00000000000E+00\t0.00000000000E+00\t0.00000000000E+00\n",
                )
                f.write(
                    "    1\t1\t0.00000000000E+00\t0.00000000000E+00\t0.00000000000E+00\t0.00000000000E+00\n",
                )
                for row in data_matrix:
                    f.write(
                        "\t%d\t%d\t%e\t%e\t%e\t%e \n"
                        % (row[0], row[1], row[2], row[3], row[4], row[5]),
                    )
            return new_name

        # https://ssd.jpl.nasa.gov/tools/gravity.html#/bennu
        # grav_20_particles.m - A MATLAB script that provides the coefficients and covariance of the estimated gravity field.
        self.sh_10 = pooch.retrieve(
            url="https://figshare.com/ndownloader/files/21927342",
            known_hash="d002eb615caab83665d991ecaa43480b85569e7f830f4aac9b2824d04d7b0dea",
            fname="Bennu_sh_10_raw.txt",
            path=os.path.dirname(GravNN.__file__) + "/Files/GravityModels/Bennu/",
            processor=format_sh,
        )

        # grav_shape_16x16.m - A MATLAB script that provides the coefficients of the shape-based uniform density gravity field.
        self.sh_shape_file = pooch.retrieve(
            url="https://figshare.com/ndownloader/files/21927351",
            known_hash="857cd603f9a9b42f60562ec19b60ac095a16005c3f1b5bd85b16436c65e5e35b",
            fname="Bennu_sh_shape_16_raw.txt",
            path=os.path.dirname(GravNN.__file__) + "/Files/GravityModels/Bennu/",
            processor=format_sh,
        )

        self.sh_file = self.sh_10


class Eros(Asteroid):
    def __init__(self):
        # Data products can be found at https://sbn.psi.edu/pds/resource/nearbrowse.html
        self.body_name = "eros"
        self.density = 2670.0  # kg/m^3 https://ssd.jpl.nasa.gov/sbdb.cgi#top
        self.physical_radius = (
            np.linalg.norm(np.array([34.4, 11.2, 11.2]) * 1e3) / 2
        )  # 34400.0/2.0# 16840.0 is *mean* diameter # meters (diameter / 2)
        self.radius = 16000.0  # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.998.2986&rep=rep1&type=pdf
        self.radius_min = 3120.0
        G = 6.67430 * 10**-11
        volume = 2525994603183.156  # m^3 from 8k file
        self.mu = G * volume * self.density

        def reindex_faces(fname, action, pooch_inst):
            "add 1 to the face indices in the obj file to work with trimesh"
            new_name = fname.split("_raw")[0] + ".obj"
            if os.path.exists(new_name):
                return new_name

            with open(fname, "r") as f:
                lines = f.readlines()

            for i in range(len(lines)):
                line = lines[i]
                if line[0] == "f":
                    lines[i] = (
                        "f "
                        + " ".join(
                            [
                                str(int(entry) + 1)
                                for entry in line.split("f")[1].split()
                            ],
                        )
                        + " \n"
                    )

            with open(new_name, "w") as f:
                f.writelines(lines)

            return new_name

        gravNN_dir = os.path.dirname(GravNN.__file__)

        # Test Models
        self.obj_66 = f"{gravNN_dir}/Files/ShapeModels/Eros/eros_shape_66.obj"
        self.obj_10k = f"{gravNN_dir}/Files/ShapeModels/Eros/eros_shape_10000.obj"

        self.obj_8k = pooch.retrieve(
            url="http://sbnarchive.psi.edu/pds3/near/NEAR_A_5_COLLECTED_MODELS_V1_0/data/msi/eros007790.tab",
            known_hash="183df4df96ea6c66dee7a4b2368dc706d81c4942fbfb043198260f5406233ff0",
            fname="eros_shape_7790_raw.obj",
            path=f"{gravNN_dir}/Files/ShapeModels/Eros/",
            processor=reindex_faces,
        )

        self.obj_90k = pooch.retrieve(
            url="http://sbnarchive.psi.edu/pds3/near/NEAR_A_5_COLLECTED_MODELS_V1_0/data/msi/eros089398.tab",
            known_hash="15184730d0a79db5d4de600fed7c758b3beb148f50d4d0e0acbebe7a1f73d82f",
            fname="eros_shape_89398_raw.obj",
            path=f"{gravNN_dir}/Files/ShapeModels/Eros/",
            processor=reindex_faces,
        )

        self.obj_200k = pooch.retrieve(
            url="http://sbnarchive.psi.edu/pds3/near/NEAR_A_5_COLLECTED_MODELS_V1_0/data/msi/eros200700.tab",
            known_hash="54c7bc73376022876a7522e002355a4046777d346fe99270c230fee92cea881f",
            fname="eros_shape_200700_raw.obj",
            path=f"{gravNN_dir}/Files/ShapeModels/Eros/",
            processor=reindex_faces,
        )

        # Spherical Harmonics
        def format_sh(fname, action, pooch_inst):
            with open(fname, "r") as f:
                data = f.read()
            processed_name = fname.split("_raw.txt")[0] + ".txt"
            with open(processed_name, "w") as f:
                f.write("    %f    %f    %f    %d\n" % (self.radius, self.mu, 0.0, 15))
                f.write(
                    "    0    0  1.00000000000E+00  0.00000000000E+00  0.00000000000E+00  0.00000000000E+00\n",
                )
                f.write(data)
            return processed_name

        self.sh_file = pooch.retrieve(
            url="http://sbnarchive.psi.edu/pds3/near/NEAR_A_5_COLLECTED_MODELS_V1_0/data/rss/n15acoeff.tab",
            known_hash="e08068e2ea5167bee685ae00a8596144964e1da71ab16c51f2328f642d0be90e",
            fname="eros_sh_N15A_raw.txt",
            path=f"{gravNN_dir}/Files/GravityModels/Eros/",
            processor=format_sh,
        )


class Toutatis(Asteroid):
    def __init__(self):
        self.body_name = "toutatis"

        G = 6.67430 * 10**-11

        # https://3d-asteroids.space/asteroids/4179-Toutatis
        self.model_lf = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Toutatis/Toutatis_Radar_based_Blender_lo_res.obj"
        )
        self.model_hf = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/ShapeModels/Toutatis/Toutatis_Radar_based_Blender_hi_res.obj"
        )
        self.obj_2k = pooch.retrieve(
            url="https://3d-asteroids.space/data/asteroids/models/t/4179_Toutatis.obj",
            known_hash="e79c13b4b7b427e3c4f90f656a257316cc1a75bc8edee36db0236129984a7f5a",
            fname="Toutatis-radar-lowres.obj",
            path=os.path.dirname(GravNN.__file__) + "/Files/ShapeModels/Toutatis/",
        )

        self.obj_20k = pooch.retrieve(
            url="https://3d-asteroids.space/data/asteroids/models/t/4179_Toutatis_hires.obj",
            known_hash="95a8dfc6aa1a75f5b96ea334d132785ec20130ce746903430e2d605bee8ed479",
            fname="Toutatis-radar-highres.obj",
            path=os.path.dirname(GravNN.__file__) + "/Files/ShapeModels/Toutatis/",
        )

        # Scheeres Paper
        # volume of 7.670 km^3
        # density = 2.5/1000.0*100**3 # kg/m^3 -- 2.5 g/cm^3 (according to Dynamics of Orbits close to Toutatis -- Scheeres)
        self.radius = 1.223 * 1000
        self.density = 2.5 / 1000.0 * 100**3
        self.mass = 1.917 * 10**13

        self.mu = G * self.mass

        # from wiki
        # self.density = 2.1/1000.0*100**3 # kg/m^3 -- 2.1 g/cm^3 (Wiki)
        # self.radius = 5.4*1E3/2 # mean diameter from https://ssd.jpl.nasa.gov/sbdb.cgi?sstr=4179
