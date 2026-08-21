import pickle
import numpy as np
from scipy.signal import savgol_filter
from data_prep.data_prep import prepare

def compute_empirical_psd(beats, seg_len=512):
    psd_sum = np.zeros(seg_len // 2 + 1, dtype=np.float64)
    count = 0
    skipped = 0
    for beat in beats:
        b = np.asarray(beat, dtype=np.float32).flatten()
        if len(b) < seg_len:
            skipped += 1
            continue
        b = b[:seg_len]
        psd_sum += np.abs(np.fft.rfft(b)) ** 2
        count += 1
    print(f"Used {count} beats, skipped {skipped} (too short)")
    if count == 0:
        raise ValueError("No valid beats found.")
    return np.fft.rfftfreq(seg_len, d=1.0 / 360), psd_sum / count


def fit_psd_anchored_savgol(freqs, h_emp, window=11, polyorder=3, anchor_freq=2.0, blend_width=3.0):
    h_emp = np.asarray(h_emp, dtype=np.float64)
    h_smooth = savgol_filter(h_emp, window_length=window, polyorder=polyorder, mode='interp')
    h_smooth = np.clip(h_smooth, 0.0, None)

    blend = np.zeros_like(freqs, dtype=np.float64)
    blend[freqs <= anchor_freq] = 1.0

    trans_mask = (freqs > anchor_freq) & (freqs <= anchor_freq + blend_width)
    if np.any(trans_mask):
        blend[trans_mask] = 0.5 * (1.0 + np.cos(np.pi * (freqs[trans_mask] - anchor_freq) / blend_width))

    h_final = blend * h_emp + (1.0 - blend) * h_smooth
    h_final = h_final / np.max(h_final)
    h_emp = h_emp / np.max(h_emp)
    h_final[np.abs(freqs - anchor_freq).argmin()] = 1.0
    return h_final, h_emp


def fit(output_path="data_prep/spectral_h.npy", seg_len=512,
        window=11, polyorder=3, anchor_freq=2.0, blend_width=3.0):

    _, y_train, _, _ = prepare()

    train_beats = [y_train[i] for i in range(len(y_train))]
    print(f"Total training beats: {len(train_beats)}")

    freqs, h_emp = compute_empirical_psd(train_beats, seg_len=seg_len)
    h_smooth, h_emp_norm = fit_psd_anchored_savgol(freqs, h_emp, window=window, polyorder=polyorder,
                                                    anchor_freq=anchor_freq, blend_width=blend_width)

    np.save(output_path, h_smooth)
    np.save(output_path.replace(".npy", "_emp.npy"), h_emp_norm)
    np.save(output_path.replace(".npy", "_freqs.npy"), freqs)

    print(f"Savitzky-Golay PSD saved to {output_path}")
    print(f"Frequency bins: {len(freqs)}, max freq: {freqs[-1]:.1f} Hz")
    return freqs, h_emp_norm, h_smooth


if __name__ == "__main__":
    fit()