import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from train_v1.inference import load_model_and_schedule, compute_metrics, compute_input_snr, run_inference
from train_v1.train import ECGDataset
from data_prep.data_prep import prepare


def plot_fig(clean, noisy, denoised, idx, snr_in, cossim, fs=250, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    t = np.arange(clean.shape[-1]) / fs

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)

    ax1.plot(t, noisy[idx], color="#D9534F", alpha=0.75, linewidth=1.0, label="Raw / Noisy ECG (Input)")
    ax1.plot(t, clean[idx], color="#2C3E50", linestyle="--", linewidth=1.2, alpha=0.8, label="Clean Reference")
    ax1.set_title(f"BEFORE DENOISING", fontsize=12, fontweight="bold", pad=8)
    ax1.set_ylabel("Amplitude (mV)", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(t, denoised[idx], color="#00A86B", linewidth=1.4, label="NEAR-SSB (Output)")
    ax2.plot(t, clean[idx], color="#2C3E50", linestyle="--", linewidth=1.2, alpha=0.8, label="Clean Reference")
    ax2.set_title(
        f"AFTER: Morphology Restored",
        fontsize=12,
        fontweight="bold",
        color="#006633",
        pad=8,
    )
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Amplitude (mV)", fontsize=10)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"demo_sample_{idx:05d}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved demo image: {save_path}")


def top_fig(
        ckpt_path="checkpoints/ma_ckpt_best_v1.pt", 
        sqrt_h_path="data_prep/ma_spectral_h.npy", 
        n_steps=1,
        batch_size=128,
        top_k=5,
        output_dir="results",
        device="cuda"
    ):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from checkpoint: {ckpt_path}")
    model, schedule = load_model_and_schedule(ckpt_path, device, sqrt_h_path)
    _, _, x_test, y_test = prepare(
        qtdb_pkl="ablation/pkl/ma_qtdb.pkl", 
        nstdb_pkl="ablation/pkl/ma_mitnoise.pkl", 
        reference_rnd_test="ablation/pkl/ma_rnd_test.npy"
    )
    test_loader = DataLoader(ECGDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    clean, noisy, denoised = run_inference(model, schedule, test_loader, device, n_steps)

    clean_ac = clean - np.mean(clean, axis=-1, keepdims=True)
    noisy_ac = noisy - np.mean(noisy, axis=-1, keepdims=True)
    denoised_ac = denoised - np.mean(denoised, axis=-1, keepdims=True)

    dot = np.sum(clean_ac * denoised_ac, axis=-1)
    norm = np.linalg.norm(clean_ac, axis=-1) * np.linalg.norm(denoised_ac, axis=-1) + 1e-8
    true_cossim = dot / norm

    sig_power = np.mean(clean_ac ** 2, axis=-1)
    noise_pwr = np.mean((noisy_ac - clean_ac) ** 2, axis=-1) + 1e-8
    true_snr_in = 10.0 * np.log10(sig_power / noise_pwr)

    valid_candidates = np.where((true_snr_in > -30.0) & (true_snr_in < -16.0))[0]
    
    if len(valid_candidates) == 0:
        valid_candidates = np.arange(len(true_cossim))

    # 5. Sort the valid candidates by highest True CosSim
    candidate_cossim = true_cossim[valid_candidates]
    best_candidate_indices = np.argsort(candidate_cossim)[-top_k:][::-1]
    top_indices = valid_candidates[best_candidate_indices]

    print("\n" + "=" * 62)
    print(f"{'RANK':<6} | {'SAMPLE ID':<10} | {'SNR IN':<10} | {'COSSIM':<10}")
    print("-" * 62)
    for rank, idx in enumerate(top_indices, start=1):
        print(
            f"#{rank:<5} | {idx:<10} | {true_snr_in[idx]:6.2f} dB | {true_cossim[idx]:.4f}"
        )
    print("=" * 62 + "\n")

    for idx in top_indices:
        plot_fig(clean, noisy, denoised, idx, true_snr_in, true_cossim, output_dir=output_dir)


if __name__ == "__main__":
    top_fig()



    