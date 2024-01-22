import os
import numpy as np
import pandas as pd
from pandas import read_csv
from scipy.fft import rfft, rfftfreq, fft, fftfreq 
from scipy.signal import find_peaks_cwt, find_peaks
from scipy.optimize import curve_fit
from functools import reduce
from itertools import accumulate
from typing import List

def read_data(fname: str):
    raw_data = read_csv(
        fname,
        skiprows=4,
        header=0,
        delimiter=',',
        index_col=0,
    )
    return raw_data

def get_list_of_files(folder: str, base_fname: str):

    try: relevant_files = list(filter(lambda fname: base_fname in fname, os.listdir(folder)))
    except FileNotFoundError: 
        print(f"ERROR: The folder '{folder}' is not in the working directory. You may need to change 'FOLDER' in main.py or download the correct data folder.")
        exit()

    if relevant_files == []:
        print(f"ERROR: Files with the common name '{base_fname}' are not found. Your '{folder}' folder may be empty or you need to change 'COMMON_FNAME' in main.py.")
        exit()

    return list(map(lambda fname: f"{folder}/{fname}", relevant_files))

def perform_fft(data):

    power = abs(rfft(data[data.columns[0]].values))
    power_db = 10 * np.log10(power/max(power))

    return power_db


def get_frequencies(data):
    time = data.index
    n_samples = len(time)
    d_time = time[1] - time[0]
    frequencies = rfftfreq(n_samples, d_time)
    return frequencies

def convert_time_to_freq_space(data):
    frequencies = get_frequencies(data)
    powers = perform_fft(data)

    frequency_data = pd.DataFrame(
        data = powers,
        index = frequencies,
        columns = [ "Power (dBm)" ],
    )
    return frequency_data

def crop_data(data):
    min = 20E6
    max = 160E6
    frequency = data.index.values
    mask = [ (freq >= min and freq <= max) for freq in frequency ]
    return data[mask]

def bin_times(data, bin_width):
    bin_times = np.arange(data.index[0], data.index[-1], bin_width)
    print("Creating edges")
    bin_edges = list(accumulate(
        bin_times[1:],
        lambda previous_edges, right_edge: [previous_edges[-1], right_edge],
        initial=[bin_times[0]]
    ))[1:]
    print("Averaging voltages...")
    average_voltages_in_bins = list(map(
        lambda bin: data.loc[(bin[0] <= data.index) & (data.index <= bin[1])].mean(),
        bin_edges
    ))
    print("Creating new times")
    new_times = list(map(
        lambda bin: (bin[1] - bin[0] / 2),
        bin_edges
    ))

    binned_data = pd.DataFrame(
        data = average_voltages_in_bins,
        index = new_times,
    )
    print("Done!")

    return binned_data

def average_over_samples(all_samples_data):
    average_frequencies = np.mean([ run.index for run in all_samples_data ], axis = 0)
    average_powers = np.mean([ run[run.columns[0]] for run in all_samples_data ], axis = 0)

    average_data = pd.DataFrame(
        data = average_powers,
        index = average_frequencies,
        columns = [ "Power (dBm)" ],
    )
    return average_data


def sum_average(data):
    return np.mean(data, axis=0)

def get_peak_locations(data):
    peaks, _ = find_peaks(data, prominence=1, width=25)
    return peaks

def flip_data(data: np.ndarray) -> np.ndarray:
    mean_value = np.mean(data)
    distances_from_mean = map( lambda datum: datum - mean_value, data )
    flipped = list(map( lambda datum, distance: datum - (2*distance), data, distances_from_mean))
    return np.array(flipped)

def number_peaks(freqs: np.ndarray, peaks: np.ndarray) -> np.ndarray:

    central_frequency = 80E6
    distances_from_centre = [ freqs[peak] - central_frequency  for peak in peaks ]
    n_of_negative_peaks = sum( 1 for d in distances_from_centre if d < 0)
    n_of_positive_peaks = sum( 1 for d in distances_from_centre if d > 0)

    numbered_peaks = [*np.arange(-1*n_of_negative_peaks, 0, 1), *np.arange(1, n_of_positive_peaks+1, 1)]

    return np.array(numbered_peaks)

def filter_n_peaks( peaks: np.ndarray, peak_number: np.ndarray, desired_n_peaks: int ) -> np.ndarray:

    peaks_to_keep = peaks[ desired_n_peaks >= abs(peak_number) ]

    return peaks_to_keep

def fit_x2(f, x: np.ndarray, y: np.ndarray):
    popt, _ = curve_fit(f, x, y)
    return popt

