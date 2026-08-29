# Train V1 Configuration

- **Diffusion Schedule**: Piecewise-linear noise schedule `g(t)` peaking at `t = 0.5` (`g_min = 1e-6`, `g_max = 1.3e-4`, `n_steps = 1000`, `eps = 1e-4`).
- **Bridge Variance**: `sigma_max = 1.0` (corrected from 0.1 to match paper standard).
- **Model Architecture**: 1D U-Net (`SpectralSBUNet`) with `base_channels = 64`, channel multipliers `(1, 2, 4, 8)`, 2 ResBlocks per level, bottleneck self-attention, and sinusoidal time embedding (`dim = 128`).
- **Model Inputs**: 2-channel concatenated input `[xt, x1]` of shape `(Batch, 2, 512)`.
- **Loss Function**: Direct MSE on clean signal `x0` (`loss = F.mse_loss(x0_pred, x0)`).
- **Optimizer & Scheduler**: Adam optimizer with `lr = 1e-3`, Cosine Annealing learning rate schedule down to `eta_min = 1e-6` over 400 epochs.
- **Early Stopping**: 50 epochs patience on validation loss.
- **Dataset**: Preloaded from `data_prep/dataset.pkl` with 70/30 train/validation split.
