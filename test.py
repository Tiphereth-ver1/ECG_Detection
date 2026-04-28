from pathlib import Path

from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from scipy.signal import iirnotch, filtfilt, sosfiltfilt, butter
from scipy.fft import fft, fftfreq
from scipy.differentiate import derivative

# Parameter control zone
fs = 100
DURATION = 500000
REFRACTORY_LOW = 30
REFRACTORY_HIGH = 200
THRESHOLD = 400
FIGSIZE = (18,8) # Control this to control the size of the output png
LOW_CUTOFF = 1
HIGH_CUTOFF = 40
# From memory, the data is approximately:
# Frequency of the ECG data is 1000 Hz from memory
# (0 - 60000) resting HR data
# (60001 - 120000) post-exercise data
# (120001 - 180000) motion artifacts
# (180001 - 210000) amplified noise

sos_low = butter(2, LOW_CUTOFF, btype='highpass', fs=fs, output='sos')
sos_high = butter(2, HIGH_CUTOFF, btype='lowpass', fs=fs, output='sos')

# Next to this script: dataSet/ProjectTrainData.mat (leading "/" would mean filesystem root)
PROJECT_TRAIN_DATA = Path(__file__).resolve().parent / "dataSet" / "ProjectTrainData.mat"

# Preprocessing of the .mat file saved from before. Can be copied at your own discretion
mat_contents = loadmat(PROJECT_TRAIN_DATA)
# print(mat_contents["ECG"].ravel())

# print(mat_contents["ECG"])
# flattened = mat_contents.ravel()*1000

# Initialise the filtering process here
b,a = iirnotch(50, 8, fs = fs)


def shitty_peak_detection(signal):
    peaks = []
    threshold = np.percentile(signal, 97)
    last_refractory = -99999
    
    for i in range(1, len(signal) - 1):
        if (
            signal[i-1] < signal[i] and
            signal[i+1] < signal[i] and
            signal[i] > threshold and
            i > last_refractory + REFRACTORY_LOW
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

def plot_hexbin(x_axis, data):
    plt.figure(figsize = FIGSIZE)
    plt.hexbin(x_axis, data)
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

def plot_2_derivatives(x_axis, filtered):
    der_1 = np.gradient(filtered, x_axis)
    der_2 = np.gradient(der_1, x_axis)
    plt.figure(figsize = FIGSIZE)
    plt.plot(x_axis, filtered, color = "black", label = "magnitude")
    # plt.plot(x_axis, der_1, color = "blue", label = "velocity")
    # plt.plot(x_axis, der_2, color = "green", label = "acceleration")
    plt.ylabel("Voltage (mV)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.title("ECG Monitor at rest")
    plt.savefig("noyo.png")
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
i=1
expert_data = loadmat(PROJECT_TRAIN_DATA)["QRSexpert"].ravel()
mat_contents = loadmat(PROJECT_TRAIN_DATA)["ECG"].ravel()
accuracies = []
for signal, expert in zip(mat_contents, expert_data):
    flattened = signal.ravel()[0:DURATION]
    expert = expert.ravel().astype(int) - 1  # QRSexpert is MATLAB 1-based; Python indices are 0-based
    expert = expert[(expert >= 0) & (expert < DURATION)]
    # print(expert)

    # print(len(flattened))
    # print(flattened)
    filtered = filtfilt(b, a, flattened)
    filtered = sosfiltfilt(sos_low, filtered)
    filtered = sosfiltfilt(sos_high, filtered)
    x_axis = np.linspace(0, DURATION/fs, len(flattened))
    peaks = np.asarray(shitty_peak_detection(filtered))
    # print(peaks)

    shared = np.intersect1d(peaks, expert)

    tol = int(0.050 * fs)  # 50 ms tolerance

    peaks = np.asarray(peaks, dtype=int)
    expert = np.asarray(expert, dtype=int)

    matched = 0
    used = np.zeros(len(peaks), dtype=bool)

    for e in expert:
        # find all peaks within tolerance window
        idx = np.where(np.abs(peaks - e) <= tol)[0]
        if len(idx) > 0:
            # choose the closest unused peak
            idx = idx[~used[idx]]
            if len(idx) > 0:
                closest = idx[np.argmin(np.abs(peaks[idx] - e))]
                used[closest] = True
                matched += 1

    accuracy = matched / len(expert) * 100 if len(expert) > 0 else 0
    accuracies.append(accuracy)

    # print(peaks)
    # print(f"Expert data: {expert}")
    print(f"Patient {i}\n__________\nShared points: {matched}\nTotal points: {len(expert)}\nAccuracy: {accuracy} %\n")
    i += 1
    # plot_filtered_figure(x_axis, filtered, peaks)
    
    # plot_hexbin(x_axis, filtered)

    # plot_double_figure_overlay(x_axis, flattened, filtered)
print(f"""Final evaluation:
Mean accuracy: {np.mean(accuracies)}
Mean STD: {np.std(accuracies)}
Worst performance: {np.min(accuracies)}%
Best performance: {np.max(accuracies)}%""")
