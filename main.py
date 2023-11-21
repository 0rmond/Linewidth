import data
import matplotlib.pyplot as plt
import plot
import numpy as np

def main():

    FOLDER = "02-10-2023"
    raw_data_files = data.get_list_of_files(FOLDER, "rawdata")
    time_and_signals = list(map(data.read_data, raw_data_files))

    #time, signal = data.read_data(FOLDER+'/'+raw_data_files[4])

    frequencies = list(map(lambda t_s: data.get_frequencies(t_s[0]), time_and_signals))
    amplitudes = list(map(lambda t_s: data.perform_fft(t_s[1]), time_and_signals))

    mean_freq = data.sum_average(frequencies)
    mean_amp = data.sum_average(amplitudes)

    cropped_freq, cropped_amp = data.crop_data(mean_freq, mean_amp)
    cropped_freq_single, cropped_amp_single = data.crop_data(frequencies[0], amplitudes[0])

    lecroy_freq, lecroy_amp = data.read_data("02-10-2023/averagefft.csv")
    cropped_lecroy_freq, cropped_lecroy_amp = data.crop_data(lecroy_freq, lecroy_amp)

    model_linewidth = np.arange(0, 200E3, 1)
    model_delS = list(map(plot.delS, model_linewidth))

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    ax1.plot(time_and_signals[0][0]*1E6, time_and_signals[0][1]*1E3, color = "black")
    ax1.set_xlabel(r"Time [$\mu s$]")
    ax1.set_ylabel(r"Voltage [$mV$]")

    ax2.plot(cropped_freq_single/1E6, cropped_amp_single, color = "blue", alpha=0.3, linewidth=1, label="n=1")
    ax2.plot(cropped_freq/1E6, cropped_amp, color = "blue", alpha = 0.6, linewidth = 1, label=f"n={len(raw_data_files)}")
    ax2.plot(cropped_lecroy_freq/1E6, cropped_lecroy_amp, color = "red", linewidth = 1, label=f"n=200")

    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel("Power [dB]")

    ax2.legend()

    ax3.plot(model_linewidth/1E3, model_delS)
    ax3.set_xlabel("Linewidth [kHz]")
    ax3.set_ylabel(r"$\Delta S$ [dB]")


    plt.show()


if __name__ == "__main__":
    main()
