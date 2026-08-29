import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from config import Config
from model import SpectralSBUNet, NoiseSchedule
from train import ECGDataset

def load_model_and_schedule(ckpt_path, device, cfg=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    c_dict = ckpt.get("config", vars(cfg or Config()))
    c = Config(**{k: v for k, v in c_dict.items() if k in Config.__annotations__})
    
    model = SpectralSBUNet(c).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    schedule = NoiseSchedule(
        sigma_max=c.sigma_max,
        g_min=c.g_min,
        g_max=c.g_max,
        num_steps=c.n_schedule_steps,
        device=str(device)
    )
    return model, schedule, c

def compute_metrics(x_clean, x_denoised):
    mae = np.max(np.abs(x_denoised - x_clean), axis=-1)
    prd = 100.0 * np.linalg.norm(x_denoised - x_clean, axis=-1) / (np.linalg.norm(x_clean, axis=-1) + 1e-8)
    ssd = np.sum((x_denoised - x_clean) ** 2, axis=-1)
    dot = np.sum(x_clean * x_denoised, axis=-1)
    norm = np.linalg.norm(x_clean, axis=-1) * np.linalg.norm(x_denoised, axis=-1) + 1e-8
    cos = dot / norm
    sig_p = np.mean(x_clean ** 2, axis=-1)
    noise_p = np.mean((x_denoised - x_clean) ** 2, axis=-1) + 1e-8
    snr_out = 10.0 * np.log10(sig_p / noise_p)
    return {"MAD": mae, "PRD": prd, "SSD": ssd, "CosSim": cos, "SNR_out": snr_out}

def compute_input_snr(x_clean, x_noisy):
    sig_p = np.mean(x_clean ** 2, axis=-1)
    noise_p = np.mean((x_noisy - x_clean) ** 2, axis=-1) + 1e-8
    return 10.0 * np.log10(sig_p / noise_p)

def print_metrics(metrics, snr_in=None):
    print("=" * 50)
    print(f"  MAD     : {metrics['MAD'].mean():.4f} +/- {metrics['MAD'].std():.4f}")
    print(f"  PRD     : {metrics['PRD'].mean():.4f} +/- {metrics['PRD'].std():.4f} %")
    print(f"  SSD     : {metrics['SSD'].mean():.4f} +/- {metrics['SSD'].std():.4f}")
    print(f"  CosSim  : {metrics['CosSim'].mean():.4f} +/- {metrics['CosSim'].std():.4f}")
    print(f"  SNR out : {metrics['SNR_out'].mean():.4f} +/- {metrics['SNR_out'].std():.4f} dB")
    if snr_in is not None:
        snr_imp = metrics["SNR_out"] - snr_in
        print(f"  SNR in  : {snr_in.mean():.4f} +/- {snr_in.std():.4f} dB")
        print(f"  SNR imp : {snr_imp.mean():.4f} +/- {snr_imp.std():.4f} dB")
    print("=" * 50)

@torch.no_grad()
def run_inference(model, schedule, loader, device, n_steps=1):
    all_clean, all_noisy, all_denoised = [], [], []
    for x1, x0 in tqdm(loader, desc="Inference", leave=False):
        x1_dev = x1.to(device)
        x0_hat = model.sample(x1_dev, schedule, n_steps=n_steps)
        all_clean.append(x0.numpy())
        all_noisy.append(x1.numpy())
        all_denoised.append(x0_hat.cpu().numpy())
    clean = np.concatenate(all_clean, axis=0).squeeze(1)
    noisy = np.concatenate(all_noisy, axis=0).squeeze(1)
    denoised = np.concatenate(all_denoised, axis=0).squeeze(1)
    return clean, noisy, denoised

def visualize(clean, noisy, denoised, indices, fs=250, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    t = np.arange(clean.shape[-1]) / fs
    for idx in indices:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(t, noisy[idx], color="tab:orange", linewidth=0.8)
        axes[0].set_title(f"Sample {idx} - Noisy Input (x1)")
        axes[1].plot(t, denoised[idx], color="tab:blue", linewidth=0.8)
        axes[1].set_title(f"Sample {idx} - Denoised Output (x0 pred)")
        axes[2].plot(t, clean[idx], color="tab:green", linewidth=0.8)
        axes[2].set_title(f"Sample {idx} - Clean Reference (x0)")
        axes[2].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{idx:05d}.png"), dpi=150)
        plt.close()

def evaluate(ckpt_path=None, cfg=None, n_steps=None, indices=(0, 1, 2, 5, 10)):
    c = cfg or Config()
    target_ckpt = ckpt_path or os.path.join(c.output_dir, "ckpt_best.pt")
    device = torch.device(c.device if torch.cuda.is_available() and c.device == "cuda" else "cpu")
    steps = n_steps if n_steps is not None else c.n_inference_steps

    model, schedule, model_cfg = load_model_and_schedule(target_ckpt, device, c)
    
    with open(os.path.join(c.data_dir, "dataset.pkl"), "rb") as f:
        x_train_full, y_train_full, x_test, y_test = pickle.load(f)

    test_loader = DataLoader(ECGDataset(x_test, y_test), batch_size=c.batch_size, shuffle=False)
    clean, noisy, denoised = run_inference(model, schedule, test_loader, device, n_steps=steps)

    metrics = compute_metrics(clean, denoised)
    snr_in = compute_input_snr(clean, noisy)
    print(f"\nEvaluation Results ({steps} steps):")
    print_metrics(metrics, snr_in)

    os.makedirs(c.results_dir, exist_ok=True)
    np.save(os.path.join(c.results_dir, "clean.npy"), clean)
    np.save(os.path.join(c.results_dir, "noisy.npy"), noisy)
    np.save(os.path.join(c.results_dir, "denoised.npy"), denoised)

    if indices:
        visualize(clean, noisy, denoised, indices, fs=c.fs, output_dir=c.results_dir)
    return metrics

if __name__ == "__main__":
    evaluate()
