import os,sys
import glob
import numpy as np
import pickle as pickle
import gc

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
from Support.GravBodyInterface import GravBodyInterface
from Trajectories.UniformDist import UniformDist
import Support.transformations as transformations
# import general simulation Support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport  # general Support file with common unit test functions
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion

# import simulation related Support
from Basilisk.simulation import spacecraftPlus, spice_interface
from Basilisk.utilities import simIncludeGravBody
from Basilisk.topLevelModules import pyswice
from Basilisk.simulation import bskLogging
from Basilisk import __path__
bskPath = __path__[0]

fileName = os.path.basename(os.path.splitext(__file__)[0])
import math as m

bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)


import numpy as np
import math
import os

from Basilisk.simulation.gravityEffector import loadGravFromFileToList
from Basilisk.utilities import simIncludeGravBody

from GravityModels.AccelerationBase import AccelerationBase

class BSK(AccelerationBase):    
    def __init__(self, body, errors, trajectory=None):
        super().__init__()
        self.body = body
        self.errors = errors
        self.configure(trajectory)
        self.load()
        pass

    def generate_full_file_directory(self):
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + str(self.errors) + "/"
        pass
    
    def compute_acc(self):
        bodyInterface = self.body
        rIC = self.trajectory.position

        '''
        This function runs through a sphere of uniformly distributed points and computes the position and velocity of the
        spacecraft across a single second.
        '''

        def computeAcceleration(posData, velData):
            rVec = []
            vVec = []
            aVec = []

            for i in range(len(velData)//2):
                vi = velData[2*i, 1:4]
                vf = velData[2*i+1, 1:4]

                ti = velData[2*i, 0]
                tf = velData[2*i+1, 0]


                a = (vf - vi)/((tf-ti)*10**-9)

                rVec.append(posData[2*i,1:4])
                vVec.append(velData[2*i,1:4])
                aVec.append(a)

            return np.array(rVec), np.array(vVec), np.array(aVec)


        simTaskName = "simTask"
        simProcessName = "simProcess"
        scSim = SimulationBaseClass.SimBaseClass()

        dynProcess = scSim.CreateNewProcess(simProcessName)
        simulationTimeStep = int(1)#macros.sec2nano(5.) # 1 ns
        dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

        scObject = spacecraftPlus.SpacecraftPlus()
        scObject.ModelTag = "spacecraftBody"
        scSim.AddModelToTask(simTaskName, scObject, None, 1)

        simIncludeGravBody.loadGravFromFile(bodyInterface.gravFile
                                        , bodyInterface.gravBody[bodyInterface.planet].spherHarm
                                        , bodyInterface.gravDeg
                                        )
        scObject.gravField.gravBodies = spacecraftPlus.GravBodyVector(list(bodyInterface.gravFactory.gravBodies.values()))


        timeInitString = "2012 MAY 1 00:28:30.0"
        spiceTimeStringFormat = '%Y %B %d %H:%M:%S.%f'
        spiceObject, epochMsg = bodyInterface.gravFactory.createSpiceInterface(bskPath +'/supportData/EphemerisData/',
                                                                timeInitString,
                                                                epochInMsgName='simEpoch')
        bodyInterface.configure()
        unitTestSupport.setMessage(scSim.TotalSim,
                                simProcessName,
                                spiceObject.epochInMsgName,
                                epochMsg)

        scSim.AddModelToTask(simTaskName, bodyInterface.gravFactory.spiceObject, None, -1)

        pyswice.furnsh_c(bodyInterface.gravFactory.spiceObject.SPICEDataPath + 'de430.bsp')  # solar system bodies
        pyswice.furnsh_c(bodyInterface.gravFactory.spiceObject.SPICEDataPath + 'naif0012.tls')  # leap second file
        pyswice.furnsh_c(bodyInterface.gravFactory.spiceObject.SPICEDataPath + 'de-403-masses.tpc')  # solar system masses
        pyswice.furnsh_c(bodyInterface.gravFactory.spiceObject.SPICEDataPath + 'pck00010.tpc')  # generic Planetary Constants Kernel

        # To set the spacecraft initial conditions, the following initial position and velocity variables are set:
        vIC = np.array([0.0, 0.0, 0.0])

        simulationTime = int(2)  # ns
        samplingTime = int(1)  # ns
        scSim.TotalSim.logThisMessage(scObject.scStateOutMsgName, samplingTime)

        scSim.InitializeSimulationAndDiscover()
        posRef = scObject.dynManager.getStateObject("hubPosition")
        velRef = scObject.dynManager.getStateObject("hubVelocity")
        for i in range(len(rIC)):
            if i % 10000 == 0:
                print("Run "+ str(i))

            posRef.setState(unitTestSupport.np2EigenVectorXd(rIC[i]))
            velRef.setState(unitTestSupport.np2EigenVectorXd(vIC))

            scSim.ConfigureStopTime(simulationTime*(i+1))
            scSim.ExecuteSimulation()
            

        posData = scSim.pullMessageLogData(scObject.scStateOutMsgName + '.r_BN_N', list(range(3)))
        velData = scSim.pullMessageLogData(scObject.scStateOutMsgName + '.v_BN_N', list(range(3)))

        posData_sphere = transformations.cart2sph(posData)
        velData_sphere = transformations.cart2sph(velData)

        bodyInterface.gravFactory.unloadSpiceKernels()
        pyswice.unload_c(bodyInterface.gravFactory.spiceObject.SPICEDataPath + 'de430.bsp')  # solar system bodies
        pyswice.unload_c(bodyInterface.gravFactory.spiceObject.SPICEDataPath + 'naif0012.tls')  # leap second file
        pyswice.unload_c(bodyInterface.gravFactory.spiceObject.SPICEDataPath + 'de-403-masses.tpc')  # solar system masses
        pyswice.unload_c(bodyInterface.gravFactory.spiceObject.SPICEDataPath + 'pck00010.tpc')  # generic Planetary Constants Kernel

        scSim.terminateSimulation()
        del scSim

        rData, vData, accData = computeAcceleration(posData_sphere[1:], velData_sphere[1:])

        return accData



def run():
    body = GravBodyInterface('earth', "/Users/johnmartin/Basilisk/SupportData/LocalGravData/GGM03S.txt", degree=10)
    rIC = UniformDist(body, body.gravBody['earth'].radEquator + 10000, 10000).load()
    accData = simulate(body, rIC, generator="SH")
    gc.collect()

    
if __name__ == "__main__":
    run()

