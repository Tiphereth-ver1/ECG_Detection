from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from scipy.signal import iirnotch, filtfilt, sosfiltfilt, butter
from scipy.fft import fft, fftfreq

# Parameter control zone
fs = 1000
START = 10500
DURATION = 5000
REFRACTORY = 200
THRESHOLD = 400
FIGSIZE = (18,8) # Control this to control the size of the output png
LOW_CUTOFF = 0.5
HIGH_CUTOFF = 40
# From memory, the data is approximately:
# Frequency of the ECG data is 1000 Hz from memory
# (0 - 60000) resting HR data
# (60001 - 120000) post-exercise data
# (120001 - 180000) motion artifacts
# (180001 - 210000) amplified noise

sos_low = butter(4, LOW_CUTOFF, btype='highpass', fs=fs, output='sos')
sos_high = butter(4, HIGH_CUTOFF, btype='lowpass', fs=fs, output='sos')


# Preprocessing of the .mat file saved from before. Can be copied at your own discretion
mat_contents = loadmat('ProjectTrainData.mat')
# print(mat_contents["ECG"].ravel())

# print(mat_contents["ECG"])
# flattened = mat_contents.ravel()*1000

# Initialise the filtering process here
b,a = iirnotch(50, 8, fs = fs)


def shitty_peak_detection(signal, threshold, refractory):
    peaks = []
    last_refractory = -99999
    
    for i in range(1, len(signal) - 1):
        if (
            signal[i-1] < signal[i] and
            signal[i+1] < signal[i] and
            signal[i] > threshold and
            i > last_refractory + refractory
        ):
            peaks.append(i)
            last_refractory = i
    return peaks    

def calculate_hr_hrv(peaks):
    rr = np.diff(peaks) / fs          # seconds
    hr = 60 / np.mean(rr)

    rr_ms = rr * 1000
    rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2))

    return float(hr), float(rmssd)

def plot_figure(x_axis, data):
    plt.figure(figsize = FIGSIZE)
    plt.plot(x_axis, data)
    plt.ylabel("Voltage (mV)")
    plt.xlabel("Time (s)")
    plt.show()


def plot_filtered_figure(x_axis, filtered, peaks):
    plt.figure(figsize = FIGSIZE)
    plt.plot(x_axis, filtered)
    plt.ylabel("Voltage (mV)")
    plt.xlabel("Time (s)")
    plt.title("ECG Monitor at rest")
    plt.scatter(x_axis[peaks], filtered[peaks], s=15, color = "black")
    plt.savefig("noyo.png")
    plt.show()

def plot_double_figures_vertical():
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = FIGSIZE)

    ax1.plot(flattened)
    ax2.plot(filtered)

    ax1.set_ylabel("Voltage (mV)")
    ax1.set_xlabel("Time (s)")

    ax2.set_ylabel("Voltage (mV)")
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("doubles_plot.png")
    plt.show()

def fourier_transform():
    N = len(flattened)
    yf = fft(filtered)
    xf = fftfreq(N, 1/fs)[:N//2]
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.xlim(right = 60)
    plt.grid()
    plt.show()

def plot_double_figure_overlay(x_axis, flattened, filtered):
    plt.figure(figsize = FIGSIZE)
    plt.plot(x_axis, flattened, color = "blue", label = "Unfiltered signal")
    plt.plot(x_axis, filtered, color = "black", label = "Filtered signal")
    plt.legend() 
    plt.ylabel("Voltage (mV)")
    plt.xlabel("Time (s)")
    plt.title("Filtered vs Unfiltered ECG Signal")
    plt.savefig("referenceheart.png")
    plt.show()


mat_contents = loadmat('ProjectTrainData.mat')["ECG"].ravel()
for content in mat_contents:
    flattened = content.ravel()[0:2000]
    # print(len(flattened))
    # print(flattened)
    filtered = filtfilt(b, a, flattened)
    filtered = sosfiltfilt(sos_low, filtered)
    filtered = sosfiltfilt(sos_high, filtered)
    x_axis = np.linspace(0, DURATION/fs, len(flattened))*100
    peaks = shitty_peak_detection(filtered, THRESHOLD, REFRACTORY)
    # plot_filtered_figure(x_axis, filtered, peaks)

    plot_double_figure_overlay(x_axis, flattened, filtered)


