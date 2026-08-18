import numpy as np
import matplotlib.pyplot as plt

def plot(freqs_path="data_prep/spectral_h_savgol_freqs.npy",
         emp_path="data_prep/spectral_h_savgol_emp.npy",
         smooth_path="data_prep/spectral_h_savgol.npy",
         output_path="data_prep/psd_fit.png"):

    freqs = np.load(freqs_path)
    h_emp = np.load(emp_path)
    h_smooth = np.load(smooth_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ECG PSD Distribution — Empirical vs Savitzky–Golay Fit", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(freqs, h_emp, color="steelblue", alpha=0.5, linewidth=1.5, label="Empirical PSD")
    ax.plot(freqs, h_smooth, color="darkgreen", linewidth=2.0, label="Savitzky–Golay Fit")
    ax.axvspan(0.5, 40.0, alpha=0.07, color="green", label="Clinical band (0.5–40 Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.set_title("Linear Scale")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([freqs[0], freqs[-1]])

    ax = axes[1]
    ax.semilogy(freqs, h_emp, color="steelblue", alpha=0.5, linewidth=1.5, label="Empirical PSD")
    ax.semilogy(freqs, h_smooth, color="darkgreen", linewidth=2.0, label="Savitzky–Golay Fit")
    ax.axvspan(0.5, 40.0, alpha=0.07, color="green", label="Clinical band (0.5–40 Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD (log)")
    ax.set_title("Log Scale")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim([freqs[0], freqs[-1]])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Plot saved to {output_path}")