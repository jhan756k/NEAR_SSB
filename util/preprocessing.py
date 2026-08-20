import numpy as np
from scipy.signal import butter, iirnotch, filtfilt
from scipy.ndimage import median_filter

def remove_baseline_wander(ecg_signal, fs, cutoff=0.5):
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(4, normalized_cutoff, btype='high')
    return filtfilt(b, a, ecg_signal)

def remove_powerline_interference(ecg_signal, fs, powerline_freq=60.0, quality_factor=30):
    nyquist = fs / 2
    notch_freq = powerline_freq / nyquist
    b, a = iirnotch(notch_freq, quality_factor)
    return filtfilt(b, a, ecg_signal)

def remove_high_frequency_noise(ecg_signal, fs, cutoff=40.0):
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(4, normalized_cutoff, btype='low')
    return filtfilt(b, a, ecg_signal)

def remove_outliers(ecg_signal, threshold=5.0):
    mean = np.mean(ecg_signal)
    std = np.std(ecg_signal)
    if std == 0:
        return ecg_signal
    z_scores = np.abs((ecg_signal - mean) / std)
    outlier_mask = z_scores > threshold
    if np.any(outlier_mask):
        smoothed = median_filter(ecg_signal, size=5)
        ecg_signal = np.where(outlier_mask, smoothed, ecg_signal)
    return ecg_signal

def normalize_amplitude(ecg_signal):
    ecg_signal = ecg_signal - np.mean(ecg_signal)
    std = np.std(ecg_signal)
    if std > 0:
        ecg_signal = ecg_signal / std
    return ecg_signal

def preprocess_signal(ecg_signal, fs):
    ecg_signal = remove_baseline_wander(ecg_signal, fs, cutoff=0.5)
    ecg_signal = remove_powerline_interference(ecg_signal, fs, powerline_freq=60.0)
    ecg_signal = remove_high_frequency_noise(ecg_signal, fs, cutoff=40.0)
    ecg_signal = remove_outliers(ecg_signal, threshold=5.0)
    ecg_signal = normalize_amplitude(ecg_signal)
    return ecg_signal