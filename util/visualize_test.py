import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from inference import load_model_and_schedule, compute_metrics, compute_input_snr, print_metrics

def visualize_denoising(model, schedule, clean_chunks, noise_chunks, device = "cuda"):
    noise_scales = [0.4, 0.8, 1.25, 1.75]
    titles = [
        "(a) 0.2 <= noise <= 0.6",
        "(b) 0.6 <= noise <= 1.0",
        "(c) 1.0 <= noise <= 1.5",
        "(d) 1.5 <= noise <= 2.0"
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharey=True)
    plt.subplots_adjust(hspace=0.4)

    for idx, (scale, ax) in enumerate(zip(noise_scales, axes)):
        noisy_chunks = np.zeros_like(clean_chunks)
        for i in range(len(clean_chunks)):
            beat_max = np.max(clean_chunks[i]) - np.min(clean_chunks[i])
            noise_max = np.max(noise_chunks[i]) - np.min(noise_chunks[i])
            ase = noise_max / (beat_max + 1e-8)
            alpha = scale / ase
            noisy_chunks[i] = clean_chunks[i] + alpha * noise_chunks[i]

        x_noisy_tensor = torch.FloatTensor(noisy_chunks).unsqueeze(1).to(device)

        with torch.no_grad():
            x_denoised_tensor = model.sample(x_noisy_tensor, schedule, n_steps=1)

        clean_flat = clean_chunks.flatten()
        noisy_flat = noisy_chunks.flatten()
        denoised_flat = x_denoised_tensor.cpu().numpy().flatten().reshape(-1)

        metrics = compute_metrics(
            clean_flat[np.newaxis, :],
            denoised_flat[np.newaxis, :]
        )

        mad = metrics["MAD"][0]
        prd = metrics["PRD"][0]
        ssd = metrics["SSD"][0]
        cos_sim = metrics["CosSim"][0]

        ax.plot(clean_flat, label="Clean", color="green", linewidth=1.0)
        ax.plot(noisy_flat, label="Noisy", color="red", linewidth=0.8)

        recon_label = (
            f"Denoised\n"
            f"MAD: {mad:.4f}\n"
            f"PRD: {prd:.4f}%\n"
            f"SSD: {ssd:.4f}\n"
            f"CosSim: {cos_sim:.4f}"
        )
        ax.plot(denoised_flat, label=recon_label, color="blue", linewidth=1.0)

        ax.set_xlabel(f"Samples\n\n{titles[idx]}", fontsize=11)
        ax.set_ylabel("Amplitude (au)", fontsize=10)
        ax.set_xlim(0, len(clean_flat))

        ax.legend(
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5), 
            borderaxespad=0,
            frameon=True,
            fontsize=10
        )
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout()
    plt.savefig("reconstruction_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "checkpoints/ckpt_best.pt"
    model, schedule = load_model_and_schedule(ckpt_path, device)

    output_dir = "data_prep"
    qt_output = os.path.join(output_dir, "qtdb.pkl")
    noise_output = os.path.join(output_dir, "mitnoise.pkl")

    with open(qt_output, "rb") as f:
        qtdb = pickle.load(f)
    with open(noise_output, "rb") as f:
        nstdb = pickle.load(f)

    bw_signals = np.array(nstdb[0])
    noise_test = bw_signals[int(bw_signals.shape[0] / 2):-1, 1]

    samples = 512
    beats = []

    for b in qtdb["sel123"]:
        b_np = np.zeros(samples)
        b_sq = np.array(b)
        init_padding = 16

        if b_sq.shape[0] > (samples - init_padding):
            continue

        b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2
        beats.append(b_np)

        if len(beats)>= 14:
            break

clean_subset = np.array(beats)

noise_subset = []
noise_index = 0

for _ in range(14):
    noise = noise_test[noise_index:noise_index + samples]
    noise_subset.append(noise)
    noise_index += samples

raw_noise_subset = np.array(noise_subset)

visualize_denoising(model, schedule, clean_subset, raw_noise_subset, device=device)