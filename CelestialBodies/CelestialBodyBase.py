from Basilisk.utilities import simIncludeGravBody
from abc import ABC, abstractmethod
import csv
class SphericalHarmonicsParams:
    degree = None
    sh_file = None
    C_lm = None
    S_lm = None

class PolyhedralParams:
    poly_file = None
    density = None

class GravityInfo:
    SH = SphericalHarmonicsParams()
    Poly = PolyhedralParams()
    mu = None

class Geometry:
    object_file = None

class CelestialBodyBase(ABC):

    body_name = None
    grav_info = GravityInfo()
    geometry = Geometry()

    def __init__(self):
        if self.grav_info.SH.sh_file is not None:
            self.loadSH(self.grav_info.SH.degree+2)
        if self.grav_info.Poly.poly_file is not None:
            pass #TODO: implement object loading file
    
    def loadSH(self, maxDeg=2):
        with open(self.grav_info.SH.sh_file, 'r') as csvfile:
            gravReader = csv.reader(csvfile, delimiter=',')
            firstRow = next(gravReader)
            clmList = []
            slmList = []
            # Currently do not take the mu and radius values provided by the gravity file.
            try:
                valCurr = int(firstRow[0])
            except ValueError:
                mu = float(firstRow[1])
                radEquator = float(firstRow[0])

            clmRow = []
            slmRow = []
            currDeg = 0
            for gravRow in gravReader:
                while int(gravRow[0]) > currDeg:
                    if (len(clmRow) < currDeg + 1):
                        clmRow.extend([0.0] * (currDeg + 1 - len(clmRow)))
                        slmRow.extend([0.0] * (currDeg + 1 - len(slmRow)))
                    clmList.append(clmRow)
                    slmList.append(slmRow)
                    clmRow = []
                    slmRow = []
                    currDeg += 1
                clmRow.append(float(gravRow[2]))
                slmRow.append(float(gravRow[3]))

            self.grav_info.SH.C_lm = clmList
            self.grav_info.SH.S_lm = slmList
            return 

