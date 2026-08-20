import pickle
import numpy as np
from util.preprocessing import preprocess_signal
from scipy.signal import welch,savgol_filter

def compute_empirical_psd(beats, fs=250, nperseg=256, noverlap=128):
    psd_sum = None
    count = 0
    for beat in beats:
        b = np.asarray(beat, dtype=np.float32)
        b = preprocess_signal(b, fs)
        if len(b) < nperseg:
            continue
        freqs, psd = welch(b, fs=fs, nperseg=nperseg, noverlap=noverlap)
        if psd_sum is None:
            psd_sum = np.zeros_like(psd)
        psd_sum += psd
        count += 1
    if count == 0 or psd_sum is None:
        raise ValueError("No valid beats found for PSD computation.")
    return freqs, psd_sum / count

def fit_psd_anchored_savgol(freqs, h_emp, window=11, polyorder=3,
                           anchor_freq=2.0, blend_width=3.0):
    h_emp = np.asarray(h_emp, dtype=np.float64)
    h_smooth = savgol_filter(h_emp, window_length=window, polyorder=polyorder, mode='interp')
    h_smooth = np.clip(h_smooth, 0.0, None)

    blend = np.zeros_like(freqs, dtype=np.float64)
    anchor_mask = freqs <= anchor_freq
    blend[anchor_mask] = 1.0

    trans_mask = (freqs > anchor_freq) & (freqs <= anchor_freq + blend_width)
    if np.any(trans_mask):
        trans_freqs = freqs[trans_mask]
        blend[trans_mask] = 0.5 * (1.0 + np.cos(
            np.pi * (trans_freqs - anchor_freq) / blend_width
        ))

    h_final = blend * h_emp + (1.0 - blend) * h_smooth

    h_final = h_final / np.max(h_final)
    h_emp = h_emp / np.max(h_emp)
    h_final[abs(freqs - anchor_freq).argmin()] = 1.0
    return h_final, h_emp

def fit(qtdb_path="data_prep/qtdb.pkl", output_path="data_prep/spectral_h_savgol.npy",
        fs=250, nperseg=256, noverlap=128, window=11, polyorder=3,
        test_set=None, anchor_freq=2.0, blend_width=3.0):
    if test_set is None:
        test_set = [
            "sel123", "sel233", "sel302", "sel307", "sel820", "sel853",
            "sel16420", "sel16795", "sele0106", "sele0121",
            "sel32", "sel49", "sel14046", "sel15814"
        ]

    with open(qtdb_path, "rb") as f:
        qtdb = pickle.load(f)

    train_beats = []
    for name, beats in qtdb.items():
        if name not in test_set:
            for b in beats:
                train_beats.append(np.array(b, dtype=np.float32))

    print(f"Total training beats: {len(train_beats)}")

    freqs, h_emp = compute_empirical_psd(train_beats, fs=fs,
                                        nperseg=nperseg, noverlap=noverlap)
    h_smooth, h_emp_norm = fit_psd_anchored_savgol(freqs, h_emp,
                                                  window=window,
                                                  polyorder=polyorder,
                                                  anchor_freq=anchor_freq,
                                                  blend_width=blend_width)

    np.save(output_path, h_smooth)
    np.save(output_path.replace(".npy", "_emp.npy"), h_emp_norm)
    np.save(output_path.replace(".npy", "_freqs.npy"), freqs)

    print(f"Savitzky-Golay PSD saved to {output_path}")
    print(f"Frequency bins: {len(freqs)}, max freq: {freqs[-1]:.1f} Hz")
    return freqs, h_emp_norm, h_smooth

if __name__ == "__main__":
    fit()