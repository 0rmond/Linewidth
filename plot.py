import data
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.optimize import curve_fit
from scipy.constants import c # speed of light
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7378313

# constants used for equation (1) in the above literature: "Laser Linewidth Measurement Based on Amplitude Difference Comparison of Coherent Envelope"
# side-comments represent the symbol in the literature

# Narrow laser-linewidth measurement using short delay self-heterodyne interferometery - Zhongan Zhao

# constants
n = 1.48 # refractive index of fibre used
L = 14 # length (in m) of delay used
TAO = L / c
INITIAL_POWER = 5

def plot_rolling_averages(sma_av_frequency_data, time_intervals):
    n_intervals = len(time_intervals)
    n_cols = 3
    n_rows = math.ceil(n_intervals / 3)

    fig, axs = plt.subplots(n_rows, n_cols)

    list(map(
        lambda data, ax: data.plot(ax=ax),
        #lambda data, ax: data.plot(data, data.index, ax=ax),
        sma_av_frequency_data,
        axs.flatten()
    ))
    
    titles = list(map(lambda interval: f"Measurement Intervals: {interval} s", time_intervals))
    list(map(
        lambda ax, ti: ax.title.set_text(ti),
        axs.flatten(),
        titles
    ))
    plt.show()



def delS(linewidth):

    def S(m, k):
        # m and k are the modes of the interference pattern. m=0 is equivalent to the first EXTREME after the central peak. |m-k| = 1
        a = (linewidth**2) * (( (c * (k+2)) / (2 * n * L) )**2)
        b = (1 - np.exp(-2*np.pi*(n*L/c)*linewidth) * np.cos(np.pi * (m + 2)))
        return a*b


    S_larger = S(1, 2)
    S_smaller = S(2, 1)
    return 10*np.log10(S_larger / S_smaller)

def plot_x2_fit( xs: np.ndarray, ys: np.ndarray, ax ):

    f = lambda x, a, c: (-1 * a * (x**2)) + c
    popt = data.fit_x2(f, xs, ys)

    xs_for_fit = np.linspace(xs[0], xs[-1], 500)
    ys_for_fit = f(xs_for_fit, *popt)

    ax.plot(xs_for_fit, ys_for_fit, linewidth=1, color = "black", marker = '')


def S1(f, linewidth):
    power_contribution = (INITIAL_POWER**2) / (4*np.pi)

    numerator = linewidth
    denominator = (f**2) + (linewidth**2)

    return power_contribution * (numerator/denominator)

def S2(f, linewidth):
    exponential_contrib = np.exp(-2 * np.pi * linewidth * TAO)
    cosine_contrib = np.cos(2 * np.pi * TAO * (f + 80E6))
    
    numerator = linewidth * np.sin(2*np.pi*TAO*(f + 80E6))
    denominator = f + 80E6
    sin_contrib = numerator / denominator

    return 1 - (exponential_contrib * (cosine_contrib + sin_contrib))

def S3(f, linewidth):
    power_contrib = (np.pi * INITIAL_POWER**2) / 2
    exponential_contrib = np.exp(-2*np.pi*linewidth*TAO)
    breakpoint()
    dirac_delta = [0 if freq != 80E6 else 1 for freq in f]

    return power_contrib * exponential_contrib * dirac_delta

def S(f, linewidth):
    s1 = S1(f, linewidth)[0]
    s2 = S2(f, linewidth)[0]
    return s1*s2# + S3(f, linewidth)

def plot_model(data, ax):

    x = data[data.columns[0]].values,
    y = data.index.values
    popt = curve_fit(
        S,
        xdata = x,
        ydata = y,
    )

    xs = np.linspace(x[0], x[-1], 1000)
    breakpoint()
    ys = S(xs, *popt)
    ax.plot(xs, ys)


