import os

import numpy as np
import trimesh

from GravNN.Support.PathTransformations import make_windows_path_posix
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase


class RandomDist(TrajectoryBase):
    def __init__(self, celestial_body, radius_bounds, points, **kwargs):
        """A distribution that samples uniformly in a spherical volume.

        Args:
            celestial_body (Celestial Body): Planet about which samples should be taken
            radius_bounds (list): range of radii from which the sample can be drawn
            points (int): number of samples
        """
        self.radius_bounds = radius_bounds
        self.points = int(points)
        self.celestial_body = celestial_body

        uniform_volume = kwargs.get("uniform_volume", False)
        if isinstance(uniform_volume, list):
            uniform_volume = uniform_volume[0]
        self.uniform_volume = uniform_volume

        self.populate_obj_file(**kwargs)
        super().__init__(**kwargs)

        pass

    def populate_obj_file(self, **kwargs):
        try:
            self.obj_file = self.celestial_body.obj_file
        except Exception:
            # asteroids obj_file is the shape model
            obj_file = kwargs.get("obj_file", [None])[0]
            if obj_file is not None:
                obj_file = make_windows_path_posix(obj_file)

            # planets have shape model (sphere currently)
            self.obj_file = kwargs.get("obj_file", [obj_file])

            try:
                # can happen if planet + asteroid in same df
                if np.isnan(self.obj_file[0]):
                    self.obj_file = [obj_file]
            except Exception:
                pass

            if isinstance(self.obj_file, list):
                self.obj_file = self.obj_file[0]

        # If the file was saved on windows but we are running on mac, load the mac path.
        self.obj_file = make_windows_path_posix(self.obj_file)

        _, file_extension = os.path.splitext(self.obj_file)
        self.filename = os.path.basename(self.obj_file)
        self.obj_mesh = trimesh.load_mesh(
            self.obj_file,
            file_type=file_extension[1:],
        )

    def generate_full_file_directory(self):
        directory_name = os.path.splitext(os.path.basename(__file__))[0]
        body = self.celestial_body.body_name
        try:
            model_name = os.path.basename(self.obj_file).split(".")[0]
        except Exception:
            model_name = str(self.obj_file.split("Blender_")[1]).split(".")[0]

        N_points = int(self.points)
        rad_bounds = self.radius_bounds
        uniform_vol = self.uniform_volume
        self.trajectory_name = f"{directory_name}/{body}_{model_name}_N_{N_points}"

        # If the first value is 0, then make sure it's saved as 0.0
        # to allow for proper loading
        bounds_str = str(rad_bounds)
        if "[0," in bounds_str:
            bounds_str = bounds_str.replace("[0,", "[0.0,")

        self.trajectory_name += f"_RadBounds{bounds_str}_UVol_{uniform_vol}"
        self.file_directory += self.trajectory_name + "/"

    def sample_volume(self, points):
        X = []
        Y = []
        Z = []
        X.extend(np.zeros((points,)).tolist())
        Y.extend(np.zeros((points,)).tolist())
        Z.extend(np.zeros((points,)).tolist())

        theta = np.random.uniform(0, 2 * np.pi, size=(points,))
        cosphi = np.random.uniform(-1, 1, size=(points,))
        R_min = self.radius_bounds[0]
        R_max = self.radius_bounds[1]

        if self.uniform_volume:
            # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
            u_min = (R_min / R_max) ** 3
            u_max = 1.0

            # want distribution to be uniform across volume the sphere
            u = np.random.uniform(u_min, u_max, size=(points,))

            # convert the uniform volume length into physical radius
            r = R_max * u ** (1.0 / 3.0)
        else:
            r = np.random.uniform(R_min, R_max, size=(points,))
        phi = np.arccos(cosphi)

        X = r * np.sin(phi) * np.cos(theta)
        Y = r * np.sin(phi) * np.sin(theta)
        Z = r * np.cos(phi)

        return np.transpose(np.array([X, Y, Z]))  # [N x 3]

    def identify_interior_points(self, positions):
        # Necessary to cap memory footprint
        N = len(positions)
        step = 500
        mask = np.full((N,), False)
        rayObject = trimesh.ray.ray_triangle.RayMeshIntersector(self.obj_mesh)
        for i in range(0, N, step):
            end_idx = (i // step + 1) * step
            position_subset = positions[i:end_idx] / 1e3
            mask[i:end_idx] = rayObject.contains_points(position_subset)
            print(i / N)
        return mask

    def assess_skip_condition(self):
        """These bodies shapes are currently spheres so there
        should be no interior points
        """
        exceptions = [
            "Earth.obj",
            "Moon.obj",
        ]
        if self.filename in exceptions:
            return True

        return False

    def recursively_remove_interior_points(self, positions):
        if self.assess_skip_condition():
            return positions

        mask = self.identify_interior_points(positions)
        interior_points = np.sum(mask)
        print(f"Remaining Points: {interior_points}")
        if interior_points > 0:
            new_positions = self.sample_volume(interior_points)
            positions[mask] = self.recursively_remove_interior_points(new_positions)
        return positions

    def generate(self):
        """Randomly sample from uniform latitude, longitude, and radial distributions

        Returns:
            np.array: cartesian positions of the samples
        """
        positions = self.sample_volume(self.points)
        positions = self.recursively_remove_interior_points(positions)
        self.positions = positions
        return positions.copy()


if __name__ == "__main__":
    from GravNN.CelestialBodies.Planets import Earth

    traj = RandomDist(Earth(), [Earth().radius, Earth().radius * 2], 10000)
