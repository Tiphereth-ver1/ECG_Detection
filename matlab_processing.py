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
THRESHOLD = 0.75
FIGSIZE = (9,8) # Control this to control the size of the output png
LOW_CUTOFF = 0.5
HIGH_CUTOFF = 40

sos_low = butter(4, LOW_CUTOFF, btype='highpass', fs=fs, output='sos')
sos_high = butter(4, HIGH_CUTOFF, btype='lowpass', fs=fs, output='sos')

# Preprocessing of the .mat file saved from before. Can be copied at your own discretion
mat_contents = loadmat('ProjectTrainData.mat')["ECG"].ravel()
for content in mat_contents:
    flattened = content.ravel()
    print(len(flattened))
    print(flattened)