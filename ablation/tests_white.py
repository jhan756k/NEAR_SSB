import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import os

from data_prep.data_prep import prepare
from ablation.inference_white import load_model_and_schedule, compute_metrics, compute_input_snr, print_metrics


def eval_segments(ckpt_path, n_steps=1, batch_size=128, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, schedule = load_model_and_schedule(ckpt_path, device)

    output_dir = "ablation/results_white"
    _, _, x_test, y_test = prepare(noise_version=1, qtdb_pkl="ablation/pkl/em_qtdb.pkl", nstdb_pkl="ablation/pkl/em_mitnoise.pkl", reference_rnd_test="ablation/pkl/em_rnd_test.npy")

    x_test = torch.FloatTensor(x_test).permute(0, 2, 1)
    y_test = torch.FloatTensor(y_test).permute(0, 2, 1)

    rnd_test_path = "ablation/pkl/em_rnd_test.npy"
    n_levels = np.load(rnd_test_path)

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(y_test), torch.FloatTensor(x_test)),
        batch_size=batch_size,
        shuffle=False
    )

    all_clean, all_noisy, all_denoised = [], [], []

    with torch.no_grad(), tqdm(test_loader, desc="Evaluating") as it:
        for clean, noisy in it:
            clean, noisy = clean.to(device), noisy.to(device)
            output = model.sample(noisy, schedule, n_steps=n_steps)

            all_clean.append(clean.cpu().numpy().reshape(clean.shape[0], -1))
            all_noisy.append(noisy.cpu().numpy().reshape(noisy.shape[0], -1))
            all_denoised.append(output.cpu().numpy().reshape(output.shape[0], -1))

    clean_full = np.concatenate(all_clean, axis=0)
    noisy_full = np.concatenate(all_noisy, axis=0)
    denoised_full = np.concatenate(all_denoised, axis=0)

    print("\n" + "=" * 50)
    print(" ALL NOISE LEVELS (OVERALL)")
    print("=" * 50)
    overall_metrics = compute_metrics(clean_full, denoised_full)
    overall_snr_in = compute_input_snr(clean_full, noisy_full)
    print_metrics(overall_metrics, overall_snr_in)

    segs = [0.2, 0.6, 1.0, 1.5, 2.0]

    for i in range(len(segs) - 1):
        idx = np.where((n_levels >= segs[i]) & (n_levels < segs[i + 1]))[0]

        print("\n" + "=" * 50)
        print(f" {segs[i]} <= noise <= {segs[i+1]}")

        if len(idx) == 0:
            print("No samples in this segment.")
            continue

        clean_subset = clean_full[idx]
        noisy_subset = noisy_full[idx]
        denoised_subset = denoised_full[idx]

        subset_metrics = compute_metrics(clean_subset, denoised_subset)
        subset_snr_in = compute_input_snr(clean_subset, noisy_subset)
        print_metrics(subset_metrics, subset_snr_in)

if __name__ == "__main__":
    eval_segments(
        ckpt_path="checkpoints/ablation_noise_artifact/ckpt_best.pt",
        n_steps=1,
        batch_size=128,
        device="cuda"
    )