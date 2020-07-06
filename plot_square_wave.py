import numpy as np
import matplotlib.pyplot as plt
from Visualization.VisualizationBase import VisualizationBase


class FourierVis(VisualizationBase):
    def __init__(self):
        super().__init__()
        self.T = 8

        pass

    def squareWave(self, x):
        lowerBoundLeft = (-self.T/2)
        lowerBoundRight = 0
        upperBoundLeft = 0
        upperBoundRight = (self.T/2)
        one = 1
        negativeOne = -1

        while True:
            if (x >= lowerBoundLeft) and (x <= lowerBoundRight):
                return negativeOne
            elif (x >= upperBoundLeft) and (x <= upperBoundRight):
                return one
            else:
                lowerBoundLeft -= self.T/2
                lowerBoundRight -= self.T/2
                upperBoundLeft += self.T/2
                upperBoundRight += self.T/2
                if one == 1:
                    one = -1
                    negativeOne = 1
                else:
                    one = 1
                    negativeOne = -1

    # Bn coefficients
    def bn(self,n):
        n = int(n)
        if (n%2 != 0):
            return 4/(np.pi*n)
        else:
            return 0

    # Wn
    def wn(self, n):
        wn = (2*np.pi*n)/self.T
        return wn

    # Fourier Series function
    def fourierSeries(self, n_max,x):
        a0 = 0
        partialSums = a0
        for n in range(1,n_max):
            try:
                partialSums = partialSums + self.bn(n)*np.sin(self.wn(n)*x)
            except:
                print("pass")
                pass
        return partialSums


y = []
f_low = []
f_mid = []
f_high = []

x = np.linspace(-20,20,10000)

vis = FourierVis()
for i in x:
    y.append(vis.squareWave(i))
    f_low.append(vis.fourierSeries(5,i))
    f_mid.append(vis.fourierSeries(15,i))
    f_high.append(vis.fourierSeries(50,i))


plt.plot(x,y,color="blue",label="Signal")
plt.plot(x,f_low,color="red",label="Fourier series approximation: 5")
plt.plot(x,f_mid,color="orange",label="Fourier series approximation: 15")
plt.plot(x,f_high,color="yellow",label="Fourier series approximation: 50")

#plt.title("Fourier Series approximation number of harmonics: "+str(vis.harmonics))
plt.legend()
plt.show()