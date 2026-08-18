# train.py (coworker code modified to use symmetric noise scheduler)
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    # Training parameters
    "num_epochs": 400,
    "batch_size": 16,
    "learning_rate": 5e-3,
    "lr_scheduler": {
        "step_size": 90,
        "gamma": 0.1,
    },

    # Symmetric noise schedule (piecewise-linear beta with peak at t=0.5)
    # We scale the integrated variance so that sigma^2(1) == sigma_max^2
    "noise_schedule": {
        "sigma_max": 1.0,   # target total variance at t=1
        "beta_min": 1e-6,
        "beta_max": 1.3e-4,
        "t_epsilon": 1e-4,  # avoid endpoints during training
    },

    # Spectral PSD H parameters - Spline-8 model (optimized from clean ECG spectrum)
    # This model uses cubic spline interpolation between control points
    "spectral_psd": {
        "type": "spline",
        "control_points": [
            0.0,
            0.09868331914813269,
            0.16901208428120226,
            0.17145097488450323,
            0.34437303860363616,
            0.34485141332153485,
            0.3459636577907434,
            0.5
        ],
        "control_values": [
            0.013306226001975174,
            0.013818607840998975,
            0.025024714532547288,
            0.012432791444381142,
            0.01379293220098221,
            0.6096359887710424,
            0.3235153792364316,
            0.04748421503894429
        ]
    },

    # Global normalization (as in coworker's code)
    "normalization": {
        "mean": -3.1605554795532953e-07,
        "std": 0.999997079372406,
    },
}


# ============================================================
# Dataset
# ============================================================

class ECGPairedDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, target_length=225000, global_mean=None, global_std=None):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.target_length = target_length
        self.global_mean = global_mean
        self.global_std = global_std

        clean_files = sorted([f.stem for f in self.clean_dir.glob("*.npy")])
        noisy_files = sorted([f.stem for f in self.noisy_dir.glob("*.npy")])
        self.paired_ids = sorted(set(clean_files) & set(noisy_files))

        if not self.paired_ids:
            raise ValueError(f"No paired files found in {clean_dir} and {noisy_dir}")

        print(f"Found {len(self.paired_ids)} paired ECG records")

    def __len__(self):
        return len(self.paired_ids)

    def __getitem__(self, idx):
        record_id = self.paired_ids[idx]

        clean_path = self.clean_dir / f"{record_id}.npy"
        noisy_path = self.noisy_dir / f"{record_id}.npy"

        clean_data = np.load(clean_path, allow_pickle=True)
        noisy_data = np.load(noisy_path, allow_pickle=True)

        if isinstance(clean_data, np.ndarray) and clean_data.dtype == object:
            clean_data = clean_data.item()
            if isinstance(clean_data, dict):
                clean_data = clean_data.get("signal", clean_data)

        if isinstance(noisy_data, np.ndarray) and noisy_data.dtype == object:
            noisy_data = noisy_data.item()
            if isinstance(noisy_data, dict):
                noisy_data = noisy_data.get("signal", noisy_data)

        clean_data = np.asarray(clean_data, dtype=np.float32)
        noisy_data = np.asarray(noisy_data, dtype=np.float32)

        if clean_data.ndim > 1:
            clean_data = clean_data.mean(axis=1)
        if noisy_data.ndim > 1:
            noisy_data = noisy_data.mean(axis=1)

        if len(clean_data) < self.target_length:
            pad = self.target_length - len(clean_data)
            clean_data = np.pad(clean_data, (0, pad), mode="edge")
        else:
            clean_data = clean_data[: self.target_length]

        if len(noisy_data) < self.target_length:
            pad = self.target_length - len(noisy_data)
            noisy_data = np.pad(noisy_data, (0, pad), mode="edge")
        else:
            noisy_data = noisy_data[: self.target_length]

        if self.global_mean is not None and self.global_std is not None:
            g_mean = float(self.global_mean)
            g_std = float(self.global_std)
            clean_data = (clean_data - g_mean) / g_std
            noisy_data = (noisy_data - g_mean) / g_std
        else:
            c_mean, c_std = clean_data.mean(), clean_data.std() + 1e-8
            clean_data = (clean_data - c_mean) / c_std
            n_mean, n_std = noisy_data.mean(), noisy_data.std() + 1e-8
            noisy_data = (noisy_data - n_mean) / n_std

        clean_tensor = torch.from_numpy(clean_data).unsqueeze(0)  # [1,L]
        noisy_tensor = torch.from_numpy(noisy_data).unsqueeze(0)  # [1,L]
        return clean_tensor, noisy_tensor


# ============================================================
# Denoising network (1D U-Net)
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DeepEcgUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim * 8),
        )

        self.enc1 = nn.Conv1d(in_channels, base_dim, 3, padding=1)
        self.down1 = nn.Conv1d(base_dim, base_dim * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(nn.SiLU(), nn.Conv1d(base_dim * 2, base_dim * 2, 3, padding=1))
        self.down2 = nn.Conv1d(base_dim * 2, base_dim * 4, 4, stride=2, padding=1)

        self.enc3 = nn.Sequential(nn.SiLU(), nn.Conv1d(base_dim * 4, base_dim * 4, 3, padding=1))
        self.down3 = nn.Conv1d(base_dim * 4, base_dim * 8, 4, stride=2, padding=1)

        self.bot = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(base_dim * 8, base_dim * 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_dim * 8, base_dim * 8, 3, padding=1),
        )

        self.up3 = nn.ConvTranspose1d(base_dim * 8, base_dim * 4, 4, stride=2, padding=1)
        self.dec3 = nn.Conv1d(base_dim * 8, base_dim * 4, 3, padding=1)

        self.up2 = nn.ConvTranspose1d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Conv1d(base_dim * 4, base_dim * 2, 3, padding=1)

        self.up1 = nn.ConvTranspose1d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.dec1 = nn.Conv1d(base_dim * 2, base_dim, 3, padding=1)

        self.out = nn.Conv1d(base_dim, out_channels, 3, padding=1)
        self.act = nn.SiLU()

    @staticmethod
    def _match_length(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        diff = ref.size(-1) - src.size(-1)
        if diff > 0:
            src = F.pad(src, (0, diff))
        elif diff < 0:
            src = src[..., : ref.size(-1)]
        return src

    def forward(self, x, t):
        t_emb = self.time_mlp(t)          # [B, base_dim*8]
        t_emb = t_emb.unsqueeze(-1)       # [B, base_dim*8, 1]

        x1 = self.act(self.enc1(x))
        x2 = self.act(self.down1(x1))
        x2 = self.enc2(x2)
        x3 = self.act(self.down2(x2))
        x3 = self.enc3(x3)
        x4 = self.act(self.down3(x3))

        x_bot = self.bot(x4 + t_emb)

        d3 = self.act(self.up3(x_bot))
        d3 = self._match_length(d3, x3)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.act(self.dec3(d3))

        d2 = self.act(self.up2(d3))
        d2 = self._match_length(d2, x2)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.act(self.dec2(d2))

        d1 = self.act(self.up1(d2))
        d1 = self._match_length(d1, x1)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.act(self.dec1(d1))

        return self.out(d1)


# ============================================================
# Symmetric scheduler + Spectral bridge (H = PSD)
# ============================================================

class SymmetricBetaSchedule:
    """
    Piecewise-linear beta(t) with peak at t=0.5:
      beta(t) increases from beta_min -> beta_max on [0,0.5]
      beta(t) decreases from beta_max -> beta_min on [0.5,1]
    We use analytic integral sigma^2(t) = ∫_0^t beta(τ)dτ.
    """
    def __init__(self, beta_min=1e-6, beta_max=1.3e-4):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def sigma2_raw(self, t: torch.Tensor) -> torch.Tensor:
        t_mid = 0.5
        slope_up = (self.beta_max - self.beta_min) / t_mid
        slope_down = (self.beta_min - self.beta_max) / t_mid

        # integral on [0,t] if t<=0.5
        sigma2_up = self.beta_min * t + 0.5 * slope_up * t**2

        # integral up to 0.5
        t0 = t_mid
        sigma2_half = self.beta_min * t0 + 0.5 * slope_up * t0**2

        # extra for t>0.5: integrate beta_max + slope_down*(τ-0.5) from 0.5..t
        dt = torch.clamp(t - t_mid, min=0.0)
        extra = self.beta_max * dt + 0.5 * slope_down * dt**2

        return torch.where(t <= t_mid, sigma2_up, sigma2_half + extra)


class SpectralI2SBBridge:
    def __init__(
        self,
        device: torch.device,
        signal_length: int,
        sigma_max: float = 1.0,
        beta_min: float = 1e-6,
        beta_max: float = 1.3e-4,
        psd_builder=None,
        psd_config: dict = None,
    ):
        self.device = device
        self.signal_length = signal_length

        self.schedule = SymmetricBetaSchedule(beta_min=beta_min, beta_max=beta_max)

        # Scale so that sigma^2(1) == sigma_max^2
        one = torch.ones((), device=device)
        S_raw = self.schedule.sigma2_raw(one)  # scalar tensor
        self.S_raw = S_raw
        self.S_target = torch.tensor(float(sigma_max**2), device=device)
        self.scale = self.S_target / (self.S_raw + 1e-20)

        # rFFT frequencies
        freqs = torch.fft.rfftfreq(signal_length, d=1.0).to(device)

        if psd_builder is None:
            if psd_config is None:
                psd_config = CONFIG["spectral_psd"]

            H = self._build_psd_filter(freqs, psd_config)
        else:
            H = psd_builder(freqs)

        self.H = torch.clamp(H, min=1e-6)  # [F]

    def _build_psd_filter(self, freqs: torch.Tensor, psd_config: dict) -> torch.Tensor:
        """
        Build PSD filter H from config. Supports both legacy 7-param and spline models.
        """
        psd_type = psd_config.get("type", "legacy")
        
        if psd_type == "spline":
            # Spline-based PSD model
            control_points = psd_config["control_points"]
            control_values = psd_config["control_values"]
            
            # Convert to numpy for scipy interpolation
            cp = np.array(control_points)
            cv = np.array(control_values)
            freqs_np = freqs.cpu().numpy()
            
            # Cubic spline interpolation
            from scipy.interpolate import CubicSpline
            spline = CubicSpline(cp, cv, bc_type='natural')
            H_np = spline(freqs_np)
            H = torch.from_numpy(H_np).float().to(self.device)
            
        else:
            # Legacy 7-param model
            H = torch.ones_like(freqs)
            f_low = float(psd_config["f_low"])
            f_mid = float(psd_config["f_mid"])
            f_high = float(psd_config["f_high"])
            psd_low = float(psd_config["psd_low"])
            psd_mid_scale = float(psd_config["psd_mid_scale"])
            psd_mid_sigma = float(psd_config["psd_mid_sigma"])
            psd_high = float(psd_config["psd_high"])

            low_mask = freqs < f_low
            high_mask = freqs > f_high
            mid_mask = (~low_mask) & (~high_mask)

            H[low_mask] = psd_low
            H[high_mask] = psd_high
            H[mid_mask] = psd_mid_scale * torch.exp(
                -((freqs[mid_mask] - f_mid) ** 2) / (2 * (psd_mid_sigma ** 2))
            )
        
        return H

    def sigma2_t(self, t: torch.Tensor) -> torch.Tensor:
        # sigma^2(t) = scaled ∫_0^t beta
        return self.scale * self.schedule.sigma2_raw(t) + 1e-8

    def sigma2_bar_t(self, t: torch.Tensor) -> torch.Tensor:
        # sigma_bar^2(t) = ∫_t^1 beta = sigma^2(1) - sigma^2(t)
        # Using scaled totals:
        return (self.S_target - self.scale * self.schedule.sigma2_raw(t)).clamp(min=1e-8)

    def get_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        B, C, L = x0.shape
        assert C == 1 and L == self.signal_length

        t_sc = t.to(self.device)  # [B]

        sigma2_t = self.sigma2_t(t_sc)         # [B]
        sigma2_bar_t = self.sigma2_bar_t(t_sc) # [B]
        denom = sigma2_t + sigma2_bar_t        # [B]

        w0 = (sigma2_bar_t / denom).view(B, 1, 1)
        w1 = (sigma2_t / denom).view(B, 1, 1)
        mu_t = w0 * x0 + w1 * x1

        sigma2_bridge = (sigma2_t * sigma2_bar_t) / denom  # [B]
        sigma_bridge = torch.sqrt(sigma2_bridge + 1e-8).view(B, 1, 1)

        eps_time = torch.randn_like(x0)
        eps_freq = torch.fft.rfft(eps_time, dim=-1, norm="ortho")

        H_psd = self.H.view(1, 1, -1)
        eps_freq_colored = eps_freq * torch.sqrt(H_psd)
        eps_time_colored = torch.fft.irfft(eps_freq_colored, n=L, dim=-1, norm="ortho")

        xt = mu_t + sigma_bridge * eps_time_colored
        return xt, eps_time_colored


# ============================================================
# Training loop
# ============================================================

def train_ssb(
    model: nn.Module,
    dataloader: DataLoader,
    config: dict = None,
    device: str = "cuda",
):
    if config is None:
        config = CONFIG

    device = torch.device(device)
    model.to(device)

    example_batch = next(iter(dataloader))[0]  # [B,1,L]
    _, _, L = example_batch.shape

    ns = config["noise_schedule"]
    bridge = SpectralI2SBBridge(
        device=device,
        signal_length=L,
        sigma_max=float(ns["sigma_max"]),
        beta_min=float(ns["beta_min"]),
        beta_max=float(ns["beta_max"]),
        psd_config=config["spectral_psd"],
    )

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr_scheduler"]["step_size"],
        gamma=config["lr_scheduler"]["gamma"],
    )

    model.train()
    eps = float(ns.get("t_epsilon", 1e-4))
    min_lr = 5e-6
    scheduler_active = True

    for epoch in range(config["num_epochs"]):
        total_loss = 0.0
        for clean_ecg, noisy_ecg in dataloader:
            clean_ecg = clean_ecg.to(device)  # X0
            noisy_ecg = noisy_ecg.to(device)  # X1
            B = clean_ecg.size(0)

            # sample t in (eps, 1-eps)
            t = torch.rand(B, device=device) * (1.0 - 2.0 * eps) + eps

            xt, _ = bridge.get_xt(clean_ecg, noisy_ecg, t)
            pred_x0 = model(xt, t)

            # 여기
            sigma2_t = bridge.sigma2_t(t)
            sigma_t = torch.sqrt(sigma2_t).view(B, 1, 1)
            target = (xt - clean_ecg) / sigma_t
            loss = F.mse_loss(pred_x0, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(dataloader), 1)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | "
            f"LR: {current_lr:.3e}"
        )
        
        # Stop scheduler when LR reaches min_lr
        if scheduler_active and current_lr <= min_lr:
            scheduler_active = False
            print(f"Learning rate reached {min_lr:.3e}, scheduler stopped.")
        
        if scheduler_active:
            scheduler.step()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(base_dir, "data", "clean")
    noisy_dir = os.path.join(base_dir, "data", "noisy")

    global_mean = CONFIG["normalization"]["mean"]
    global_std = CONFIG["normalization"]["std"]

    dataset = ECGPairedDataset(
        clean_dir,
        noisy_dir,
        target_length=225000,
        global_mean=global_mean,
        global_std=global_std,
    )
    print(f"Loaded {len(dataset)} paired ECG records")

    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    model = DeepEcgUNet(in_channels=1, out_channels=1, base_dim=32)

    print("\nStarting Spectral Schrödinger Bridge training (symmetric scheduler)...")
    train_ssb(model, loader, config=CONFIG, device=device)

    model_save_path = os.path.join(base_dir, "meca.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    print("Training complete.")
