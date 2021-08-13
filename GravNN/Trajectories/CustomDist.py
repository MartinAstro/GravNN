import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import numpy as np
from copy import deepcopy
import re
def read_ephemeris_head(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    object = re.split(r" {2,}",lines[1])[1]
    for i in range(len(lines)):
        if "Start time" in lines[i]:
            start_time = lines[i].split(":")[1].replace(" ", "-")
        if "Stop  time" in lines[i]:
            stop_time = lines[i].split(":")[1].replace(" ", "-")

    return object, start_time, stop_time



def read_ephemeris_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    for i in range(len(lines)):
        if "$$SOE" in lines[i]:
            start = i + 1
            break
    
    for i in range(len(lines)-1, 0, -1):
        if "$$EOE" in lines[i]:
            end = i - 1
        
    x_vec = np.zeros((end-start, 3))
    t_vec = np.zeros((end-start, 1))
    
    for k in range(start, end):
        line = lines[k]
        entries = line.split(',')
        t_vec[k-start] = float(entries[0])
        x_vec[k-start][0] = float(entries[2])
        x_vec[k-start][1] = float(entries[3])
        x_vec[k-start][2] = float(entries[4])

    return t_vec, x_vec




class CustomDist(TrajectoryBase):
    def __init__(self, ephemeris_file, max_data_entries=None, **kwargs):
        self.ephemeris_file = ephemeris_file
        self.file_name = os.path.basename(ephemeris_file).split('.')[0]
        self.object, self.start_time, self.stop_time = read_ephemeris_head(ephemeris_file)
        self.max_data_entries = max_data_entries
        super().__init__(**kwargs)

        pass

    def generate_full_file_directory(self):
        self.data_name =  self.file_name + \
                            "_start_" + str(self.start_time) + \
                            "_stop_" + str(self.stop_time) + \
                            "_N_" + str(self.max_data_entries)
        self.trajectory_name =  os.path.splitext(os.path.basename(__file__))[0] +  "/" + self.data_name
                                               
        self.file_directory  += self.trajectory_name +  "/"
        pass
    
    def generate(self):
        self.times, self.positions = read_ephemeris_data(self.ephemeris_file)
        self.dt = self.times[1] - self.times[0]

        if self.max_data_entries is not None:
            self.times = self.times[:self.max_data_entries]
            self.positions = self.positions[:self.max_data_entries]

        self.points = len(self.times)
        return self.positions

    def __sub__(self, other):
        # Difference the positions of the two ephermii

        # Ensure that the trajectories share the same timestamps
        assert np.all(other.times == self.times)

        newEphemeris = deepcopy(self)
        newEphemeris.positions -= other.positions
        newEphemeris.file_directory += other.data_name+ "/" 

        return newEphemeris


def main():
    import matplotlib.pyplot as plt
    from GravNN.Visualization.VisualizationBase import VisualizationBase
    near_traj = EphemerisDist("GravNN/Files/Ephmerides/near_positions.txt", override=False)
    eros_traj = EphemerisDist("GravNN/Files/Ephmerides/eros_positions.txt", override=True)

    near_m_eros_traj = near_traj - eros_traj
    X = near_m_eros_traj.positions[1000:,0]
    Y = near_m_eros_traj.positions[1000:,1]
    Z = near_m_eros_traj.positions[1000:,2]
    vis = VisualizationBase()
    fig, ax = vis.new3DFig()
    ax.plot3D(X, Y, Z, 'gray')
    # plt.scatter(X, Y, Z)
    plt.show()

def main2():
    import matplotlib.pyplot as plt
    from GravNN.Visualization.VisualizationBase import VisualizationBase

    import spiceypy as spice
    spice.tkvrsn("TOOLKIT")
    spice.furnsh("GravNN/Files/Ephmerides/NEAR/near_eros.txt")

    utc = ['March 02, 2000', 'Feb 02, 2001']

    # get et values one and two, we could vectorize str2et
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])

    step = 4000
    times = [x*(etTwo-etOne)/step + etOne for x in range(step)]

    positions, lightTimes = spice.spkpos('NEAR', times, 'J2000', 'NONE', 'EROS')
    print(positions)
    vis = VisualizationBase()
    fig, ax = vis.new3DFig()
    ax.plot3D(positions[:,0], positions[:,1], positions[:,2], 'gray')

    # https://www.jhuapl.edu/Content/techdigest/pdf/V23-N01/23-01-Holdridge.pdf -- NEAR Orbits
    months = ['Feb', 'March', 'April', 'May', 'June', 'July']
    for i in range(0, len(months)-1):
            
        utc = [months[i] + ' 01, 2000', months[i+1]+' 01 2000']
        times = 100000
        # get et values one and two, we could vectorize str2et
        etOne = spice.str2et(utc[0])
        etTwo = spice.str2et(utc[1])

        times = [x*(etTwo-etOne)/step + etOne for x in range(step)]
        positions, lightTimes = spice.spkpos('NEAR', times, 'EROS_FIXED', 'NONE', 'EROS')
        fig, ax = vis.new3DFig()
        ax.plot3D(positions[:,0], positions[:,1], positions[:,2], 'gray')

    plt.show()
    spice.kclear()



def main3():
    import matplotlib.pyplot as plt
    from GravNN.Visualization.VisualizationBase import VisualizationBase

    import spiceypy as spice
    spice.tkvrsn("TOOLKIT")
    spice.furnsh("GravNN/Files/Ephmerides/NEAR/near_eros.txt")


    orbits = ['Feb 24, 2000', 
            'Mar 03, 2000',
            'Apr 02, 2000',
            'Apr 11, 2000',
            'Apr 22, 2000',
            'Apr 30, 2000',
            'July 07, 2000',
            'July 14, 2000',
            'July 24, 2000',
            'July 31, 2000',
            'Aug 08, 2000',
            'Aug 26, 2000',
            'Sep 05, 2000',
            'Oct 13, 2000',
            'Oct 20, 2000',
            'Oct 25, 2000',
            'Oct 26, 2000',
            'Nov 03, 2000',
            'Dec 07, 2000',
            'Dec 13, 2000',
            'Jan 24, 2001',
            'Jan 28, 2001',
            'Jan 29, 2001',
            'Feb 02, 2001',
            'Feb 06, 2001']


    # get et values one and two, we could vectorize str2et

    vis = VisualizationBase()

    # https://www.jhuapl.edu/Content/techdigest/pdf/V23-N01/23-01-Holdridge.pdf -- NEAR Orbits
    for i in range(0, len(orbits)-1):
            
        utc = [orbits[i], orbits[i+1]]

        etOne = spice.str2et(utc[0])
        etTwo = spice.str2et(utc[1])

        step = int((etTwo - etOne)/(10*60)) # Every 10 minutes

        times = [x*(etTwo-etOne)/step + etOne for x in range(step)]
        positions, lightTimes = spice.spkpos('NEAR', times, 'EROS_FIXED', 'NONE', 'EROS')
        fig, ax = vis.new3DFig()
        ax.plot3D(positions[:,0], positions[:,1], positions[:,2], 'gray')

    plt.show()
    spice.kclear()
if __name__ == "__main__":
    main()