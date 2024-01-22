import data
import math
import matplotlib.pyplot as plt
import plot
import numpy as np
from itertools import zip_longest, repeat
from typing import List
import pandas as pd

def pair_up(xs: List):
    evens = xs[::2]
    odds = xs[1::2]
    return zip_longest(evens, odds)


def calc_init_iloc_range(data, time_interval):

    t = data.index.values
    del_t = t[1] - t[0]

    return int(math.ceil(time_interval / del_t))

def rolling_average_data(data, time_interval):
    window_size = calc_init_iloc_range(data, time_interval)
    windows = data.rolling(window_size).mean()

    sma_data = windows.iloc[window_size - 1 :]

    return sma_data

def get_frequency_data(time_data):

    frequency_data = (time_data
        .pipe(data.convert_time_to_freq_space)
        .pipe(data.crop_data)
    )
    return frequency_data

def get_sma_of_times(samples, time_interval):
    sma = list(map(
        lambda sample_data: rolling_average_data(sample_data, time_interval),
        samples
    ))
    return sma

def get_frequency_from_sma_t(samples_times, time_intervals):
    """
    The returned list will look like:
        [each time interval
            [each sample (~20)
            ],
        ]
    """
    samples_sma_data = map(
        lambda time_interval: get_sma_of_times(samples_times, time_interval),
        time_intervals
    )

    samples_sma_f_data = list(map(
        lambda intervals_samples_sma_data: list(map(
            get_frequency_data,
            intervals_samples_sma_data
            )),
        samples_sma_data
    ))

    return samples_sma_f_data



def main():

    # FFT ON DATA FROM MULTIPLE COLLECTIONS/RUNS #
    FOLDER = "02-10-2023"
    COMMON_FNAME = "rawdata"
    raw_data_files = data.get_list_of_files(FOLDER, COMMON_FNAME)

    samples_t_data = list(map(data.read_data, raw_data_files))
    samples_f_data = list(map(get_frequency_data, samples_t_data))

    samples_av_f_data = data.average_over_samples(samples_f_data)


    # FFT ON DATA THAT HAS ALREADY BEEN AVERAGED BY LECROY #

    lecroy_data = (
        data.read_data(f"{FOLDER}/averagefft.csv")
        .pipe(data.crop_data)
    )


    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    
    # UNALTERED PLOT #

    single_run_time = samples_t_data[0]

    single_run_time.plot(
        ax = ax1,
        color = "black",
    )

    ax1.set_xlabel(r"Time [$s$]")
    ax1.set_ylabel(r"Voltage [$V$]")
    ax1.set_ylim(bottom = -0.05, top = 0.3)

    # FFT PLOT #

    n_samples = len(samples_f_data)
    single_run_freq = samples_f_data[0]

    single_run_freq.plot(
        ax = ax2,
        color = "blue",
        alpha = 0.25,
    )

    samples_av_f_data.plot(
        ax = ax2,
        color = "blue",
        alpha = 0.75,
    )

    lecroy_data.plot(
        ax = ax2,
        color = "red",
        alpha = 1,
    )

    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel("Power [dB]")
    ax2.legend([
        "n samples=1",
        f"n samples={n_samples}",
        f"n_samples=200"
    ])

    # del S plot #
    model_linewidth = np.arange(0, 200E3, 1)
    model_delS = list(map(plot.delS, model_linewidth))
    ax3.plot(model_linewidth/1E3, model_delS)
    ax3.set_xlabel("Linewidth [kHz]")
    ax3.set_ylabel(r"$\Delta S$ [dB]")

    plt.show()


if __name__ == "__main__":
    main()
