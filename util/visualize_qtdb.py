from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import wfdb

data_dir = Path(__file__).resolve().parents[1] / "dataset" / "qtdb"
name = "sel872"
lead_name = "ECG1"

record = wfdb.rdrecord(str(data_dir / name))
print(f"Record: {name}")
print("Available channels:", record.sig_name)

if lead_name not in record.sig_name:
    raise ValueError(f"Lead '{lead_name}' not found in record '{name}'. Available: {record.sig_name}")

channel_index = record.sig_name.index(lead_name)
channel_name = record.sig_name[channel_index]
signal = record.p_signal[:, channel_index]
time = np.arange(len(signal)) / record.fs

print(f"Selected lead: {channel_name}")
print(f"Sampling rate: {record.fs} Hz")
print(f"Samples: {len(signal)}")

plt.figure(figsize=(12, 4))
plt.plot(time, signal, color="tab:blue", linewidth=1)
plt.title(f"QTDB ECG waveform: {name} / {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
