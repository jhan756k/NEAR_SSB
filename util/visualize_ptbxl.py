from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import ast
from scipy import signal
from scipy.ndimage import median_filter

data_dir = Path(__file__).resolve().parents[1] / "dataset" / "ptbxl"
ecg_id = 200  # Set to specific ECG ID, or None to manually select
preprocess = True  # Set to False to visualize raw signal

def remove_baseline_wander(ecg_signal, fs, cutoff=0.5):
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='high')
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal

def remove_powerline_interference(ecg_signal, fs, powerline_freq=60.0, quality_factor=30):
    nyquist = fs / 2
    notch_freq = powerline_freq / nyquist
    b, a = signal.iirnotch(notch_freq, quality_factor)
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal

def remove_high_frequency_noise(ecg_signal, fs, cutoff=40.0):
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal

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

# Load database metadata
db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
db.scp_codes = db.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load diagnostic superclass mapping
agg_df = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def get_diagnostic_superclass(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

db["diagnostic_superclass"] = db.scp_codes.apply(get_diagnostic_superclass)

# Select record
if ecg_id is None:
    print("Available ECG IDs:")
    for i, eid in enumerate(db.index[:20]):
        print(f"  {eid}: Patient {db.loc[eid, 'patient_id']}, Age {db.loc[eid, 'age']}")
    print("  ...")
    ecg_id = int(input("\nSelect ECG ID: "))

if ecg_id not in db.index:
    raise ValueError(f"ECG ID {ecg_id} not found in database")
record_path = db.loc[ecg_id, "filename_hr"]
record = wfdb.rdsamp(str(data_dir / record_path))

ecg_data, meta = record
fs = meta["fs"]

print(f"\nECG ID: {ecg_id}")
print(f"Patient ID: {db.loc[ecg_id, 'patient_id']}")
print(f"Age: {db.loc[ecg_id, 'age']}")
print(f"Sex: {db.loc[ecg_id, 'sex']}")
print(f"Sampling rate: {fs} Hz")
print(f"Duration: {len(ecg_data) / fs:.2f} seconds")
print(f"Channels: {meta['sig_name']}")
print(f"Diagnostic superclass: {db.loc[ecg_id, 'diagnostic_superclass']}")

# Apply preprocessing if enabled
if preprocess:
    for i in range(ecg_data.shape[1]):
        ecg_data[:, i] = preprocess_signal(ecg_data[:, i], fs)

# Plot leads in 2 columns
n_leads = ecg_data.shape[1]
time = np.arange(len(ecg_data)) / fs
n_rows = (n_leads + 1) // 2

fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2 * n_rows))
axes = axes.flatten()

for i in range(n_leads):
    ax = axes[i]
    ax.plot(time, ecg_data[:, i], color="tab:blue", linewidth=0.5)
    ax.set_ylabel(meta["sig_name"][i], fontsize=10)
    ax.grid(True, alpha=0.3)

# Hide the extra subplot if odd number of leads
if n_leads % 2 != 0:
    axes[-1].set_visible(False)

for i in range(n_leads):
    if i >= n_leads - 2:  # Only show x-label on bottom plots
        axes[i].set_xlabel("Time (s)")

title_suffix = " (Preprocessed)" if preprocess else " (Raw)"
fig.suptitle(f"PTB-XL ECG (ID: {ecg_id}) - {db.loc[ecg_id, 'diagnostic_superclass']}{title_suffix}", fontsize=12)
plt.tight_layout()
plt.show()
