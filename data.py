import os
import numpy as np
from pandas import read_csv
from scipy.fft import rfft, rfftfreq, fft, fftfreq 
from functools import reduce

def read_data(fname):
    raw_data = read_csv(fname, skiprows=4,delimiter=',')
    return [ raw_data["Time"].values, raw_data["Ampl"].values ]

def get_list_of_files(folder, base_fname):
    relevant_files = list(filter(lambda fname: base_fname in fname, os.listdir(folder)))
    return list(map(lambda fname: f"{folder}/{fname}", relevant_files))

def perform_fft(signal):
    ampl = np.array(rfft(signal))
    return 10 * np.log10(ampl/max(ampl))

def get_frequencies(time):
    n_samples = len(time)
    d_time = time[1] - time[0]
    return np.array(rfftfreq(n_samples, d_time))

def crop_data(frequency, amplitude):
    min = 20E6
    max = 160E6
    mask = [ (freq >= min and freq <= max) for freq in frequency ]
    return [frequency[mask], amplitude[mask]]

def sum_average(data):
    return np.mean(data, axis=0)
