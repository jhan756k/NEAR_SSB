import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

CONFIG = {
    "num_epochs": 400,
    "batch_size": 16,
    "learning_rate": 5e-3,
    "lr_scheduler": {"step_size": 90, "gamma": 0.1},
    "min_lr": 5e-6,

    "signal": {
        "fs": 250,
        "samples": 512,
        "init_padding": 16,
    },

    "noise_schedule": {
        "sigma_max": 1.0,
        "beta_min": 1e-6,
        "beta_max": 1.3e-4,
        "t_epsilon": 1e-4,
    },

    "noise_injection": {
        "target_snr_db": 0.0,
        "alpha_low": 0.2,
        "alpha_high": 2.0,
    },

    "paths": {
        "qtdb_pkl":     "data_prep/qtdb.pkl",
        "nstdb_pkl":    "data_prep/mitnoise.pkl",
        "spectral_h":   "data_prep/spectral_h.npy",
        "spectral_freqs": "data_prep/spectral_h_freqs.npy",
        "model_out":    "near_ssb.pt",
    },
}

TEST_SET = [
    "sel123", "sel233", "sel302", "sel307", "sel820", "sel853", "sel16420",
    "sel16795", "sele0106", "sele0121", "sel32", "sel49", "sel14046", "sel15814"
]

class ECGBeatDataset(Dataset):
    def __init__(self, qtdb, noise_signal, cfg):
        self.cfg    = cfg
        self.noise  = noise_signal.astype(np.float32)
        self.beats  = []

        samples      = cfg["signal"]["samples"]
        init_padding = cfg["signal"]["init_padding"]

        for name, beat_list in qtdb.items():
            if name in TEST_SET:
                continue
            for b in beat_list:
                b_np = np.array(b, dtype=np.float32)
                if b_np.shape[0] > (samples - init_padding):
                    continue
                padded = np.zeros(samples, dtype=np.float32)
                padded[init_padding:b_np.shape[0] + init_padding] = (
                    b_np - (b_np[0] + b_np[-1]) / 2.0
                )
                self.beats.append(padded)

        print(f"Training beats: {len(self.beats)}")

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        samples = self.cfg["signal"]["samples"]
        x0 = self.beats[idx].copy()

        noise_start = (idx * samples) % max(1, len(self.noise) - samples)
        noise_seg   = self.noise[noise_start: noise_start + samples]
        if len(noise_seg) < samples:
            noise_seg = np.pad(noise_seg, (0, samples - len(noise_seg)), mode="wrap")

        p_x0    = np.mean(x0 ** 2) + 1e-12
        p_noise = np.mean(noise_seg ** 2) + 1e-12
        snr_db  = self.cfg["noise_injection"]["target_snr_db"]
        p_target = p_x0 / (10 ** (snr_db / 10.0))
        n_scaled = noise_seg * np.sqrt(p_target / p_noise)

        alpha_low  = self.cfg["noise_injection"]["alpha_low"]
        alpha_high = self.cfg["noise_injection"]["alpha_high"]
        alpha = np.random.uniform(alpha_low, alpha_high)
        x1 = x0 + alpha * n_scaled

        x0_t = torch.from_numpy(x0).unsqueeze(0)
        x1_t = torch.from_numpy(x1).unsqueeze(0)
        return x0_t, x1_t

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb  = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class UNet1D(nn.Module):
    def __init__(self, base_dim=64):
        super().__init__()
        bd = base_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(bd),
            nn.Linear(bd, bd * 4),
            nn.SiLU(),
            nn.Linear(bd * 4, bd * 8),
        )

        # Encoder
        self.enc1  = nn.Conv1d(1,      bd,     3, padding=1)
        self.down1 = nn.Conv1d(bd,     bd * 2, 4, stride=2, padding=1)
        self.enc2  = nn.Sequential(nn.SiLU(), nn.Conv1d(bd * 2, bd * 2, 3, padding=1))
        self.down2 = nn.Conv1d(bd * 2, bd * 4, 4, stride=2, padding=1)
        self.enc3  = nn.Sequential(nn.SiLU(), nn.Conv1d(bd * 4, bd * 4, 3, padding=1))
        self.down3 = nn.Conv1d(bd * 4, bd * 8, 4, stride=2, padding=1)

        # Bottleneck — time injected here
        self.bot = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(bd * 8, bd * 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(bd * 8, bd * 8, 3, padding=1),
        )

        # Decoder
        self.up3  = nn.ConvTranspose1d(bd * 8, bd * 4, 4, stride=2, padding=1)
        self.dec3 = nn.Conv1d(bd * 8, bd * 4, 3, padding=1)
        self.up2  = nn.ConvTranspose1d(bd * 4, bd * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Conv1d(bd * 4, bd * 2, 3, padding=1)
        self.up1  = nn.ConvTranspose1d(bd * 2, bd,     4, stride=2, padding=1)
        self.dec1 = nn.Conv1d(bd * 2, bd,     3, padding=1)

        self.out  = nn.Conv1d(bd, 1, 3, padding=1)
        self.act  = nn.SiLU()

    @staticmethod
    def _match(src, ref):
        d = ref.size(-1) - src.size(-1)
        if d > 0:
            src = F.pad(src, (0, d))
        elif d < 0:
            src = src[..., :ref.size(-1)]
        return src

    def forward(self, x, t):
        t_emb = self.time_mlp(t).unsqueeze(-1)     # [B, bd*8, 1]

        x1 = self.act(self.enc1(x))
        x2 = self.act(self.down1(x1))
        x2 = self.enc2(x2)
        x3 = self.act(self.down2(x2))
        x3 = self.enc3(x3)
        x4 = self.act(self.down3(x3))

        xb = self.bot(x4 + t_emb)

        d3 = self.act(self.up3(xb))
        d3 = self._match(d3, x3)
        d3 = self.act(self.dec3(torch.cat([d3, x3], dim=1)))

        d2 = self.act(self.up2(d3))
        d2 = self._match(d2, x2)
        d2 = self.act(self.dec2(torch.cat([d2, x2], dim=1)))

        d1 = self.act(self.up1(d2))
        d1 = self._match(d1, x1)
        d1 = self.act(self.dec1(torch.cat([d1, x1], dim=1)))

        return self.out(d1)

class SymmetricSchedule:
    def __init__(self, sigma_max, beta_min, beta_max, device):
        self.beta_min  = beta_min
        self.beta_max  = beta_max
        self.device    = device
        self.sigma_max = sigma_max

        one     = torch.ones(1, device=device)
        raw_one = self._raw(one)
        self.c  = (sigma_max ** 2) / raw_one.item()

    def _raw(self, t):
        t_mid      = 0.5
        slope_up   = (self.beta_max - self.beta_min) / t_mid
        slope_down = (self.beta_min - self.beta_max) / t_mid

        sig2_up = self.beta_min * t + 0.5 * slope_up * t ** 2

        sig2_half = self.beta_min * t_mid + 0.5 * slope_up * t_mid ** 2

        dt    = torch.clamp(t - t_mid, min=0.0)
        extra = self.beta_max * dt + 0.5 * slope_down * dt ** 2

        return torch.where(t <= t_mid, sig2_up, sig2_half + extra)

    def sigma2(self, t):
        return self.c * self._raw(t) + 1e-8

    def sigma2_bar(self, t):
        return (self.sigma_max ** 2 - self.c * self._raw(t)).clamp(min=1e-8)

class SpectralBridge:
    def __init__(self, schedule, h, signal_length, device):
        self.schedule = schedule
        self.L        = signal_length
        self.device   = device

        self.h = h.to(device)

    def sample_xt(self, x0, x1, t):
        B, _, L = x0.shape

        s2    = self.schedule.sigma2(t).view(B, 1, 1)
        s2bar = self.schedule.sigma2_bar(t).view(B, 1, 1)
        denom = s2 + s2bar

        w0   = s2bar / denom
        w1   = s2    / denom
        mu_t = w0 * x0 + w1 * x1

        sigma_bridge = torch.sqrt((s2 * s2bar) / denom)

        eps      = torch.randn(B, 1, L, device=self.device)
        eps_freq = torch.fft.fft(eps, dim=-1)

        h_sqrt        = torch.sqrt(self.h).view(1, 1, L)
        eps_freq_col  = eps_freq * h_sqrt
        eps_colored   = torch.fft.ifft(eps_freq_col, dim=-1).real

        xt = mu_t + sigma_bridge * eps_colored
        return xt, eps_colored

def build_h_from_fitted_psd(h_smooth, freqs_welch, signal_length, fs, device):

    from scipy.interpolate import interp1d

    interp_fn = interp1d(
        freqs_welch, h_smooth,
        kind="linear", bounds_error=False,
        fill_value=(h_smooth[0], h_smooth[-1])
    )

    dft_freqs = np.fft.fftfreq(signal_length, d=1.0 / fs) 
    dft_freqs_abs = np.abs(dft_freqs)

    h_full = interp_fn(dft_freqs_abs).astype(np.float32)
    h_full = h_full / (h_full.max() + 1e-12)

    return torch.from_numpy(h_full).to(device)

def train(cfg=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(cfg["paths"]["qtdb_pkl"], "rb") as f:
        qtdb = pickle.load(f)
    with open(cfg["paths"]["nstdb_pkl"], "rb") as f:
        nstdb = pickle.load(f)

    # Use BW noise channel 1a for training (consistent with prep_data.py)
    bw_signals   = np.array(nstdb[0])
    noise_train  = bw_signals[:bw_signals.shape[0] // 2, 0].astype(np.float32)

    h_path     = cfg["paths"]["spectral_h"]
    freqs_path = cfg["paths"]["spectral_freqs"]

    h_smooth    = np.load(h_path).astype(np.float32)
    freqs_welch = np.load(freqs_path).astype(np.float32)

    L  = cfg["signal"]["samples"]
    fs = cfg["signal"]["fs"]
    h  = build_h_from_fitted_psd(h_smooth, freqs_welch, L, fs, device)

    dataset = ECGBeatDataset(qtdb, noise_train, cfg)
    loader  = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = UNet1D(base_dim=64).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    ns       = cfg["noise_schedule"]
    schedule = SymmetricSchedule(
        sigma_max=ns["sigma_max"],
        beta_min=ns["beta_min"],
        beta_max=ns["beta_max"],
        device=device,
    )
    bridge = SpectralBridge(schedule, h, L, device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["lr_scheduler"]["step_size"],
        gamma=cfg["lr_scheduler"]["gamma"],
    )

    eps_t         = ns["t_epsilon"]
    min_lr        = cfg["min_lr"]
    sched_active  = True

    model.train()
    for epoch in range(cfg["num_epochs"]):
        total_loss = 0.0

        for x0, x1 in loader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            B  = x0.size(0)

            t = torch.rand(B, device=device) * (1.0 - 2.0 * eps_t) + eps_t

            xt, _ = bridge.sample_xt(x0, x1, t)
            pred   = model(xt.detach(), t)
            sigma_t = torch.sqrt(bridge.schedule.sigma2(t)).view(B, 1, 1)
            target  = (xt - x0) / sigma_t

            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss   = total_loss / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.3e}")

        if sched_active:
            scheduler.step()
            if optimizer.param_groups[0]["lr"] <= min_lr:
                sched_active = False
                print(f"LR floor {min_lr:.1e} reached — scheduler stopped.")

    torch.save(model.state_dict(), cfg["paths"]["model_out"])
    print(f"Model saved to {cfg['paths']['model_out']}")


if __name__ == "__main__":
    train()