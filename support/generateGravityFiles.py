import numpy as np
import os
import random
from Basilisk import __path__
bskPath = __path__[0]

def main():
    originalGrav = bskPath + '/supportData/LocalGravData/GGM03S.txt'
    newFile = ""

    dist = "UniformTrue"
    if not os.path.exists("/Users/johnmartin/Documents/GraduateSchool/Research/nnGravCoef/Files/gravFiles/"+dist + "/"):
        os.makedirs("/Users/johnmartin/Documents/GraduateSchool/Research/nnGravCoef/Files/gravFiles/" + dist + "/")

    for m in range(1001, 2001):
        newFile = ""
        with open(originalGrav, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                newLine = ""
                if i == 0:
                    newLine = line
                else:
                    words = line.split(',')


                    for k in range(len(words)):
                        value = float(words[k])
                        if k < 2:
                            newLine += str(int(value)) + ",\t"
                        elif k > 3:
                            newLine += str(0) + ",\t" # no uncertainty (to reduce file size)
                        else:
                            if i > 0 and i < 4:
                                newLine += str(value) + ",\t" # C00 - C11 are all unchanged -- (point mass and dipole)
                            else:
                                if dist == "Uniform":
                                    randomValue = random.uniform(-1, 1)*0.25 # up to 25% error
                                    newLine += str(value + value*randomValue) + ",\t"
                                elif dist == "UniformTrue":
                                    newLine += str(random.uniform(-1, 1)) + ",\t"
                                else:
                                    newLine += str(random.gauss(0, 1E-6)) + ",\t"

                    newLine += "\n"

                newFile += str(newLine)


        with open("//Users/johnmartin/Documents/GraduateSchool/Research/nnGravCoef/Files/gravFiles/"+dist+"/GMM_"+str(m)+".txt", 'w') as f:
            f.write(newFile)

if __name__ == '__main__':
    main()