import os


class Earth:
    def __init__(self):
        self.body_name = "earth"
        self.mu = 0.3986004415e15  # meters^3/s^2
        self.radius = 6378136.6  # meters
        self.sh_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/GravityModels/GGM03S.txt"
        )
        self.sh_hf_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/GravityModels/EGM2008_to2190_TideFree_E.txt"
        )


class Moon:
    def __init__(self):
        self.body_name = "moon"
        self.radius = 1738100.0  # meters
        self.mu = 4.902799e12  # meters^3/s^2
        self.sh_hf_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../Files/GravityModels/gggrx_1200a_sha.txt"
        )
