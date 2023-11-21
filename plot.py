import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c # speed of light
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7378313

# constants used for equation (1) in the above literature: "Laser Linewidth Measurement Based on Amplitude Difference Comparison of Coherent Envelope"
# side-comments represent the symbol in the literature

# Narrow laser-linewidth measurement using short delay self-heterodyne interferometery - Zhongan Zhao

# constants
n = 1.48 # refractive index of fibre used
L = 14 # length (in m) of delay used

def delS(linewidth):

    def S(m, k):
        # m and k are the modes of the interference pattern. m=0 is equivalent to the first EXTREME after the central peak. |m-k| = 1
        a = (linewidth**2) * (( (c * (k+2)) / (2 * n * L) )**2)
        b = (1 - np.exp(-2*np.pi*(n*L/c)*linewidth) * np.cos(np.pi * (m + 2)))
        return a*b


    S_larger = S(1, 2)
    S_smaller = S(2, 1)
    return 10*np.log10(S_larger / S_smaller)


