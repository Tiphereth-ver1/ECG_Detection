from scipy import stats
from scipy import signal
from astropy.timeseries import LombScargle
import numpy as np

sample_rr_peak_interval_data = [1, 2, 3, 4, 5, 7 ,6]  # Example data


rr_interval_array = np.array(sample_rr_peak_interval_data)
time_stamps = np.cumsum(rr_interval_array)  # They really chose the worst name for this command  


avr_rr_int = np.mean(rr_interval_array)

detrended_rr = rr_interval_array - np.mean(rr_interval_array)

stdev_rr_int = np.std(data, ddof=1)

rmssd  = np.sqrt(np.mean(np.diff(rr_interval_array)**2))

succ_rr_50ms = (np.sum(np.abs(np.diff(rr_interval_array)) > 0.05) / (len(rr_interval_array) - 1)) * 100

LS = LombScargle(time_stamps, detrended_rr)

min_f_LF = 0.04
max_f_LF = 0.15
freq_LF = np.linspace(min_f_LF, max_f_LF, 1000)  
power = LS.power(freq_LF, normalization='psd')
LF_band = (freq_LF >= min_f_LF) & (freq_LF <= max_f_LF)
LF_power = np.trapezoid(power[LF_band], freq_LF[LF_band])

min_f_HF = 0.15
max_f_HF = 0.4
freq_HF = np.linspace(min_f_HF, max_f_HF, 1000)
power = LS.power(freq_HF, normalization='psd')
HF_band = (freq_HF >= min_f_HF) & (freq_HF <= max_f_HF)
HF_power = np.trapezoid(power[HF_band], freq_HF[HF_band])

print(avr_rr_int)
print(stdev_rr_int)
print(rmssd)
print(LF_power)
print(HF_power)
