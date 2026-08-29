import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from config import Config
from model import SpectralSBUNet, NoiseSchedule
from train import ECGDataset
from data_prep.data_prep import prepare
from inference import load_model_and_schedule, compute_metrics, compute_input_snr, print_metrics

def eval_noise_segments(ckpt_path=None, cfg=None, n_steps=None):
    c = cfg or Config()
    target_ckpt = ckpt_path or os.path.join(c.output_dir, "ckpt_best.pt")
    device = torch.device(c.device if torch.cuda.is_available() and c.device == "cuda" else "cpu")
    steps = n_steps if n_steps is not None else c.n_inference_steps

    model, schedule, model_cfg = load_model_and_schedule(target_ckpt, device, c)
    _, _, x_test, y_test = prepare(
        qtdb_pkl=c.qtdb_pkl,
        nstdb_pkl=c.nstdb_pkl,
        reference_rnd_test=c.reference_rnd_test,
        output_dir=c.data_dir
    )

    test_loader = DataLoader(ECGDataset(x_test, y_test), batch_size=c.batch_size, shuffle=False)
    all_clean, all_noisy, all_denoised = [], [], []

    for x1, x0 in tqdm(test_loader, desc="Testing", leave=False):
        x1_dev = x1.to(device)
        x0_hat = model.sample(x1_dev, schedule, n_steps=steps)
        all_clean.append(x0.numpy())
        all_noisy.append(x1.numpy())
        all_denoised.append(x0_hat.cpu().numpy())

    clean_full = np.concatenate(all_clean, axis=0).squeeze(1)
    noisy_full = np.concatenate(all_noisy, axis=0).squeeze(1)
    denoised_full = np.concatenate(all_denoised, axis=0).squeeze(1)

    print("\nOVERALL TEST METRICS:")
    overall_metrics = compute_metrics(clean_full, denoised_full)
    overall_snr_in = compute_input_snr(clean_full, noisy_full)
    print_metrics(overall_metrics, overall_snr_in)

    if os.path.exists(c.reference_rnd_test):
        n_levels = np.load(c.reference_rnd_test)
        segs = [0.2, 0.6, 1.0, 1.5, 2.0]
        for i in range(len(segs) - 1):
            idx = np.where((n_levels >= segs[i]) & (n_levels < segs[i + 1]))[0]
            print(f"\nNOISE RANGE [{segs[i]} <= noise < {segs[i+1]}]:")
            if len(idx) == 0:
                print("No samples in this range.")
                continue
            sub_metrics = compute_metrics(clean_full[idx], denoised_full[idx])
            sub_snr_in = compute_input_snr(clean_full[idx], noisy_full[idx])
            print_metrics(sub_metrics, sub_snr_in)

if __name__ == "__main__":
    eval_noise_segments()
