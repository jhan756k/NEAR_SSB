import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Configuration (MUST match the fixed symmetric-schedule train.py)
# ============================================================

CONFIG = {
    "num_steps":2,
    "target_length": 225000,

    # Symmetric schedule params (match train)
    "noise_schedule": {
        "sigma_max": 1.0,     # same meaning as in fixed train: sigma^2(1)=sigma_max^2 after scaling
        "beta_min": 1e-6,
        "beta_max": 1.3e-4,
        "t_epsilon": 1e-4,
    },

    # PSD H(ω) (must match train) - Spline-8 model
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
    # Global normalization (must match train)
    "normalization": {
        "mean": -3.1605554795532953e-07,
        "std": 0.999997079372406,
    },

    "base_dim": 32,

    # Sampling behavior
    "eta": 0.0,          # 0.0 = deterministic (recommended); >0 adds colored noise each step
    "clamp_x0": None,    # e.g. 5.0 if outputs explode
}


# ============================================================
# Model (same as in train file)
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
        t_emb = self.time_mlp(t).unsqueeze(-1)  # (B, base_dim*8, 1)

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
# Symmetric beta schedule (same math as fixed train)
# ============================================================

class SymmetricBetaSchedule:
    def __init__(self, beta_min=1e-6, beta_max=1.3e-4):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def sigma2_raw(self, t: torch.Tensor) -> torch.Tensor:
        """
        sigma^2_raw(t) = ∫_0^t beta(τ)dτ for a piecewise-linear beta with peak at 0.5.
        (Unscaled; we scale it so sigma^2(1) matches sigma_max^2.)
        """
        t_mid = 0.5
        slope_up = (self.beta_max - self.beta_min) / t_mid
        slope_down = (self.beta_min - self.beta_max) / t_mid

        sigma2_up = self.beta_min * t + 0.5 * slope_up * t**2

        t0 = t_mid
        sigma2_half = self.beta_min * t0 + 0.5 * slope_up * t0**2

        dt = torch.clamp(t - t_mid, min=0.0)
        extra = self.beta_max * dt + 0.5 * slope_down * dt**2

        return torch.where(t <= t_mid, sigma2_up, sigma2_half + extra)


def _bridge_stats(schedule: SymmetricBetaSchedule, t: torch.Tensor, sigma_max: float, scale: torch.Tensor):
    """
    Returns (w0, w1, std) for:
      x_t = w0*x0 + w1*x1 + std*z
    with:
      sigma^2(t)     = scale * sigma2_raw(t)
      sigma_bar^2(t) = sigma_max^2 - sigma^2(t)
      std^2          = (sigma^2*sigma_bar^2)/(sigma^2+sigma_bar^2)
    """
    # t: (B,)
    s2 = (scale * schedule.sigma2_raw(t)).clamp(min=1e-12)
    s2_bar = (sigma_max * sigma_max - s2).clamp(min=1e-12)
    denom = (s2 + s2_bar).clamp(min=1e-20)

    w0 = (s2_bar / denom).view(-1, 1, 1)
    w1 = (s2 / denom).view(-1, 1, 1)

    var = (s2 * s2_bar) / denom
    std = torch.sqrt(torch.clamp(var, min=1e-20)).view(-1, 1, 1)
    return w0, w1, std


# ============================================================
# Bridge sampler (PSD H + symmetric schedule)
# ============================================================

class SpectralBridgeSampler:
    """
    Holds PSD H(ω) and the symmetric schedule scaling constant.
    Also provides colored noise generation in the same way as training.
    """
    def __init__(self, device, signal_length, config):
        self.device = device
        self.signal_length = signal_length

        ns = config["noise_schedule"]
        self.sigma_max = float(ns["sigma_max"])
        beta_min = float(ns["beta_min"])
        beta_max = float(ns["beta_max"])

        self.schedule = SymmetricBetaSchedule(beta_min=beta_min, beta_max=beta_max)

        # Scale so sigma^2(1) == sigma_max^2 (same as fixed train)
        one = torch.tensor(1.0, device=device)
        S_raw = self.schedule.sigma2_raw(one)  # scalar tensor
        self.scale = torch.tensor((self.sigma_max ** 2), device=device) / (S_raw + 1e-20)

        # Build PSD H(ω) as in training
        psd_cfg = config["spectral_psd"]
        freqs = torch.fft.rfftfreq(signal_length, d=1.0).to(device)

        H = self._build_psd_filter(freqs, psd_cfg)
        self.H_psd = torch.clamp(H, min=1e-6)  # [F]

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

    def colored_noise_time(self, shape):
        """
        Generates eps_time_colored via: eps_freq * sqrt(H), then irfft back.
        Matches your coworker training/inference style.
        """
        B, C, L = shape
        eps_time = torch.randn(shape, device=self.device)
        eps_freq = torch.fft.rfft(eps_time, dim=-1, norm="ortho")
        H_sqrt = torch.sqrt(self.H_psd).view(1, 1, -1)
        eps_freq_colored = eps_freq * H_sqrt
        eps_time_colored = torch.fft.irfft(eps_freq_colored, n=L, dim=-1, norm="ortho")
        return eps_time_colored


# ============================================================
# Reverse-time sampler (DDIM-like bridge sampler; symmetric schedule)
# ============================================================

@torch.no_grad()
def sample_ssb_reverse(model, x1_noisy, sampler: SpectralBridgeSampler, num_steps, t_epsilon, eta=0.0, clamp_x0=None):
    """
    Deterministic by default (eta=0):
      1) x0_hat = model(x_t, t)
      2) infer z_hat from current bridge equation
      3) set x_{t_next} = w0_next*x0_hat + w1_next*x1 + std_next*z_use

    If eta>0, mixes fresh colored noise each step:
      z_use = (1-eta)*z_hat + eta*z_fresh
    """
    device = x1_noisy.device
    model.eval()

    x_t = x1_noisy.clone()
    x1 = x1_noisy.clone()

    B = x_t.size(0)
    times = torch.linspace(1.0, float(t_epsilon), int(num_steps), device=device)

    for i in range(len(times) - 1):
        t_curr = float(times[i].item())
        t_next = float(times[i + 1].item())

        t_curr_b = torch.full((B,), t_curr, device=device)
        t_next_b = torch.full((B,), t_next, device=device)

        # predict (xt - x0)/sigma_t and recover x0
        pred = model(x_t, t_curr_b)
        sigma2_t = (sampler.scale * sampler.schedule.sigma2_raw(t_curr_b)).clamp(min=1e-12)
        sigma_t = torch.sqrt(sigma2_t).view(B, 1, 1)
        x0_hat = x_t - sigma_t * pred
        if clamp_x0 is not None:
            x0_hat = x0_hat.clamp(-clamp_x0, clamp_x0)

        # current stats
        w0_c, w1_c, std_c = _bridge_stats(
            sampler.schedule, t_curr_b, sigma_max=sampler.sigma_max, scale=sampler.scale
        )

        # z_hat (safe at t=1 where std ~ 0)
        z_hat = (x_t - (w0_c * x0_hat + w1_c * x1)) / std_c
        z_hat = torch.where(std_c < 1e-10, torch.zeros_like(z_hat), z_hat)

        # next stats
        w0_n, w1_n, std_n = _bridge_stats(
            sampler.schedule, t_next_b, sigma_max=sampler.sigma_max, scale=sampler.scale
        )

        if eta == 0.0:
            z_use = z_hat
        else:
            z_fresh = sampler.colored_noise_time(x_t.shape)
            z_use = (1.0 - eta) * z_hat + eta * z_fresh

        x_t = w0_n * x0_hat + w1_n * x1 + std_n * z_use

    # final prediction at smallest t
    t_final = torch.full((B,), float(times[-1].item()), device=device)
    pred = model(x_t, t_final)
    sigma2_t = (sampler.scale * sampler.schedule.sigma2_raw(t_final)).clamp(min=1e-12)
    sigma_t = torch.sqrt(sigma2_t).view(B, 1, 1)
    x0_hat = x_t - sigma_t * pred
    if clamp_x0 is not None:
        x0_hat = x0_hat.clamp(-clamp_x0, clamp_x0)
    return x0_hat


# ============================================================
# Main I/O
# ============================================================

if __name__ == "__main__":
    MODEL_PATH = "meca.pt"
    NOISY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test", "noisy")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "denoised_output")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    print(f"Loading model from: {model_path}")
    model = DeepEcgUNet(in_channels=1, out_channels=1, base_dim=CONFIG["base_dim"]).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully!")

    # Files
    noisy_dir = Path(NOISY_DIR)
    noisy_files = sorted(noisy_dir.glob("*.npy"))
    print(f"\nFound {len(noisy_files)} noisy ECG files in: {NOISY_DIR}")
    if not noisy_files:
        raise SystemExit("No .npy files found!")

    signal_length = int(CONFIG["target_length"])
    sampler = SpectralBridgeSampler(device=device, signal_length=signal_length, config=CONFIG)

    num_steps = int(CONFIG["num_steps"])
    ns = CONFIG["noise_schedule"]
    t_eps = float(ns.get("t_epsilon", 1e-4))
    eta = float(CONFIG.get("eta", 0.0))
    clamp_x0 = CONFIG.get("clamp_x0", None)

    GLOBAL_MEAN = float(CONFIG["normalization"]["mean"])
    GLOBAL_STD = float(CONFIG["normalization"]["std"])

    print(f"\nRunning SSB inference with {num_steps} reverse steps (eta={eta})...")

    for idx, noisy_path in enumerate(noisy_files, 1):
        record_id = noisy_path.stem

        noisy_obj = np.load(noisy_path, allow_pickle=True)
        if isinstance(noisy_obj, np.ndarray) and noisy_obj.dtype == object:
            noisy_obj = noisy_obj.item()
        if isinstance(noisy_obj, dict):
            noisy_signal = noisy_obj.get("signal", noisy_obj)
        else:
            noisy_signal = noisy_obj

        noisy_signal = np.asarray(noisy_signal, dtype=np.float32)

        if noisy_signal.ndim > 1:
            noisy_signal = noisy_signal.mean(axis=1)

        if len(noisy_signal) < signal_length:
            pad_size = signal_length - len(noisy_signal)
            noisy_signal = np.pad(noisy_signal, (0, pad_size), mode="edge")
        else:
            noisy_signal = noisy_signal[:signal_length]

        # Global normalization (must match training)
        noisy_norm = (noisy_signal - GLOBAL_MEAN) / (GLOBAL_STD + 1e-12)
        x1_noisy = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device)

        # Reverse sample -> predict x0
        x0_hat = sample_ssb_reverse(
            model=model,
            x1_noisy=x1_noisy,
            sampler=sampler,
            num_steps=num_steps,
            t_epsilon=t_eps,
            eta=eta,
            clamp_x0=clamp_x0,
        )

        den = x0_hat.squeeze().detach().cpu().numpy()
        den = den * GLOBAL_STD + GLOBAL_MEAN

        out_path = os.path.join(OUTPUT_DIR, f"{record_id}_denoised.npy")
        np.save(out_path, den.astype(np.float32))

        print(f"[{idx}/{len(noisy_files)}] Processed {record_id}: {den.shape}")

    print(f"\nAll denoised data saved to: {OUTPUT_DIR}")
    print("Done.")
