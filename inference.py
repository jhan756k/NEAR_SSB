import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model import SpectralSBUNet, NoiseSchedule
from train import ECGDataset
from data_prep.data_prep import prepare


def load_model_and_schedule(ckpt_path, device, sqrt_h_path="data_prep/spectral_h.npy"):
    model = SpectralSBUNet(sqrt_h_path=sqrt_h_path).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if "schedule_config" in ckpt:
        schedule = NoiseSchedule(device=str(device), **ckpt["schedule_config"])
    else:
        # Old checkpoint saved before schedule_config was tracked. Falling back
        # to NoiseSchedule() defaults here is exactly what caused the sigma_max
        # mismatch bug (1.0 vs the 0.1 actually used in train.py) -- so warn
        # loudly instead of silently producing garbage denoised output.
        print(
            "WARNING: checkpoint has no 'schedule_config'. Falling back to "
            "NoiseSchedule() defaults, which will NOT match training unless "
            "you pass matching sigma_max/g_min/g_max/num_steps explicitly. "
            "Re-train (or re-save this checkpoint) to embed schedule_config."
        )
        schedule = NoiseSchedule(device=str(device))
    return model, schedule


def compute_metrics(x_clean, x_denoised):
    mae = np.max(np.abs(x_denoised - x_clean), axis=-1)
    prd = 100.0 * np.linalg.norm(x_denoised - x_clean, axis=-1) / (np.linalg.norm(x_clean, axis=-1) + 1e-8)
    ssd = np.sum((x_denoised - x_clean) ** 2, axis=-1)
    dot = np.sum(x_clean * x_denoised, axis=-1)
    norm = np.linalg.norm(x_clean, axis=-1) * np.linalg.norm(x_denoised, axis=-1) + 1e-8
    cos = dot / norm
    signal_power = np.mean(x_clean ** 2, axis=-1)
    noise_power = np.mean((x_denoised - x_clean) ** 2, axis=-1) + 1e-8
    snr_out = 10.0 * np.log10(signal_power / noise_power)
    return {"MAD": mae, "PRD": prd, "SSD": ssd, "CosSim": cos, "SNR_out": snr_out}


def compute_input_snr(x_clean, x_noisy):
    signal_power = np.mean(x_clean ** 2, axis=-1)
    noise_power = np.mean((x_noisy - x_clean) ** 2, axis=-1) + 1e-8
    return 10.0 * np.log10(signal_power / noise_power)


def print_metrics(metrics, snr_in=None):
    print(f"\n{'='*50}")
    print(f"  MAD     : {metrics['MAD'].mean():.4f} ± {metrics['MAD'].std():.4f}")
    print(f"  PRD     : {metrics['PRD'].mean():.4f} ± {metrics['PRD'].std():.4f} %")
    print(f"  SSD     : {metrics['SSD'].mean():.4f} ± {metrics['SSD'].std():.4f}")
    print(f"  CosSim  : {metrics['CosSim'].mean():.4f} ± {metrics['CosSim'].std():.4f}")
    print(f"  SNR out : {metrics['SNR_out'].mean():.4f} ± {metrics['SNR_out'].std():.4f} dB")
    if snr_in is not None:
        snr_imp = metrics["SNR_out"] - snr_in
        print(f"  SNR in  : {snr_in.mean():.4f} ± {snr_in.std():.4f} dB")
        print(f"  SNR imp : {snr_imp.mean():.4f} ± {snr_imp.std():.4f} dB")
    print(f"{'='*50}\n")


@torch.no_grad()
def run_inference(model, schedule, loader, device, n_steps=50):
    all_clean = []
    all_noisy = []
    all_denoised = []
    progress = tqdm(loader, desc="Inference", unit="batch")
    for x1, x0 in progress:
        x1 = x1.to(device)
        x0_hat = model.sample(x1, schedule, n_steps=n_steps)
        all_clean.append(x0.numpy())
        all_noisy.append(x1.cpu().numpy())
        all_denoised.append(x0_hat.cpu().numpy())
    clean = np.concatenate(all_clean, axis=0).squeeze(1)
    noisy = np.concatenate(all_noisy, axis=0).squeeze(1)
    denoised = np.concatenate(all_denoised, axis=0).squeeze(1)
    return clean, noisy, denoised


def visualize(clean, noisy, denoised, indices, fs=250, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    t = np.arange(clean.shape[-1]) / fs

    for idx in tqdm(indices, desc="Saving sample plots", unit="sample"):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Sample {idx}", fontsize=13)

        axes[0].plot(t, noisy[idx], color="tab:orange", linewidth=0.8)
        axes[0].set_title("Noisy Input (x1)")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(t, denoised[idx], color="tab:blue", linewidth=0.8)
        axes[1].set_title("Denoised Output (x0 predicted)")
        axes[1].set_ylabel("Amplitude")

        axes[2].plot(t, clean[idx], color="tab:green", linewidth=0.8)
        axes[2].set_title("Clean Reference (x0)")
        axes[2].set_ylabel("Amplitude")
        axes[2].set_xlabel("Time (s)")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{idx:05d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        tqdm.write(f"saved {save_path}")


def visualize_overlay(clean, noisy, denoised, indices, fs=250, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    t = np.arange(clean.shape[-1]) / fs

    for idx in tqdm(indices, desc="Saving overlay plots", unit="sample"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
        fig.suptitle(f"Sample {idx}", fontsize=13)

        axes[0].plot(t, noisy[idx], color="tab:orange", linewidth=0.8, label="Noisy")
        axes[0].plot(t, clean[idx], color="tab:green", linewidth=0.8, label="Clean")
        axes[0].set_title("Input vs Clean")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()

        axes[1].plot(t, denoised[idx], color="tab:blue", linewidth=0.8, label="Denoised")
        axes[1].plot(t, clean[idx], color="tab:green", linewidth=0.8, label="Clean")
        axes[1].set_title("Denoised vs Clean")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"overlay_{idx:05d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        tqdm.write(f"saved {save_path}")


def evaluate(
    ckpt_path,
    sqrt_h_path="data_prep/spectral_h.npy",
    n_steps=50,
    batch_size=32,
    device="cuda",
    visualize_indices=None,
    output_dir="results",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model, schedule = load_model_and_schedule(ckpt_path, device, sqrt_h_path)

    x_train, y_train, x_test, y_test = prepare()
    test_loader = DataLoader(ECGDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    print(f"Running inference on {len(x_test)} test samples with {n_steps} steps...")
    clean, noisy, denoised = run_inference(model, schedule, test_loader, device, n_steps)

    metrics = compute_metrics(clean, denoised)
    snr_in = compute_input_snr(clean, noisy)
    print_metrics(metrics, snr_in)

    np.save(os.path.join(output_dir, "clean.npy"), clean)
    np.save(os.path.join(output_dir, "noisy.npy"), noisy)
    np.save(os.path.join(output_dir, "denoised.npy"), denoised)

    if visualize_indices is not None:
        visualize(clean, noisy, denoised, visualize_indices, output_dir=output_dir)
        visualize_overlay(clean, noisy, denoised, visualize_indices, output_dir=output_dir)

    return clean, noisy, denoised, metrics


if __name__ == "__main__":
    evaluate(
        ckpt_path="checkpoints/ckpt_epoch0010.pt",
        n_steps=1,
        batch_size=64,
        device="cuda",
        visualize_indices=[0, 1, 2, 10, 50, 100],
        output_dir="results",
    )