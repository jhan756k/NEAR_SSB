import numpy as np
import os

def white_h(
        output_path="ablation/ablation_spectral_h/bw_white_spectral_h.npy",
        seg_len=512,
        fs=250
):
    freqs = np.fft.rfftfreq(seg_len, d=1/fs)
    h_white = np.ones_like(freqs, dtype=np.float32)

    output_dir = os.path.dirname(output_path)

    np.save(output_path, freqs)

    print("=" * 50)
    print("WHITE SPECTRAL H GENERATED")
    print("=" * 50)
    print(f"Output:         {output_path}")
    print(f"Segment length: {seg_len}")
    print(f"Sampling rate:  {fs} Hz")
    print(f"Frequency bins: {len(freqs)}")
    print(f"Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    print(f"H min:          {h_white.min():.1f}")
    print(f"H max:          {h_white.max():.1f}")
    print("=" * 50)

if __name__ == "__main__":
    white_h(output_path=("ablation/ablation_spectral_h/bw_white_spectral_h.npy"), seg_len=512, fs=250)    