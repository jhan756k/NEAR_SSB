import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def visualize_model_comparison(
    indices, 
    fs=250, 
    spectral_dir="results", 
    white_dir="ablation/results_white", 
    output_dir="results/comparison_plots"
):

    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    try:
        clean = np.load(os.path.join(spectral_dir, "clean.npy"))
        denoised_spectral = np.load(os.path.join(spectral_dir, "denoised.npy"))
        denoised_white = np.load(os.path.join(white_dir, "denoised.npy"))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure you have run the evaluate() function for both models first.")
        return

    t = np.arange(clean.shape[-1]) / fs

    for idx in tqdm(indices, desc="Saving comparison plots", unit="sample"):
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle(f"Model Comparison - Sample {idx}", fontsize=14, fontweight='bold')

        # Plot Clean Reference[cite: 1]
        ax.plot(t, clean[idx], color="tab:green", linewidth=1.5, alpha=0.6, label="Clean Reference (x0)")
        
        # Plot Gaussian Noise Ablation (White Noise)
        ax.plot(t, denoised_white[idx], color="tab:orange", linewidth=1.0, alpha=0.8, label="Denoised (Gaussian Noise)")
        
        # Plot Spectral H Output[cite: 1]
        ax.plot(t, denoised_spectral[idx], color="tab:blue", linewidth=1.0, alpha=0.8, label="Denoised (Spectral H)")

        ax.set_title("Spectral H vs. Gaussian Noise Overlay")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"comparison_overlay_{idx:05d}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        tqdm.write(f"Saved {save_path}")

if __name__ == "__main__":
    sample_indices_to_plot = [0, 1, 2, 10, 50, 100] 
    
    visualize_model_comparison(
        indices=sample_indices_to_plot
    )