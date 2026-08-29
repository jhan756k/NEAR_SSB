import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalEmbedding(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        return self.mlp(t)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_groups=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels * 2))
        self.norm2 = nn.GroupNorm(num_groups, out_channels, affine=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.t_proj(t_emb).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None]) + shift[:, :, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, channels) * 0.02)

    def forward(self, x):
        h = self.norm(x).permute(0, 2, 1)
        L = h.shape[1]
        h = h + self.pos_embed[:, :L, :]
        h, _ = self.attn(h, h, h)
        return x + self.proj_out(h.permute(0, 2, 1))

class UNet1D(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, t_emb_dim=128, num_groups=8, dropout=0.0):
        super().__init__()
        t_dim = t_emb_dim * 4
        self.t_embedding = TimestepEmbedding(t_emb_dim)

        self.in_proj = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        self.enc1_1 = ResBlock(base_channels, base_channels, t_dim, num_groups, dropout)
        self.enc1_2 = ResBlock(base_channels, base_channels, t_dim, num_groups, dropout)
        self.down1 = nn.Conv1d(base_channels, base_channels, 3, stride=2, padding=1)

        self.enc2_1 = ResBlock(base_channels, base_channels * 2, t_dim, num_groups, dropout)
        self.enc2_2 = ResBlock(base_channels * 2, base_channels * 2, t_dim, num_groups, dropout)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)

        self.enc3_1 = ResBlock(base_channels * 2, base_channels * 4, t_dim, num_groups, dropout)
        self.enc3_2 = ResBlock(base_channels * 4, base_channels * 4, t_dim, num_groups, dropout)
        self.down3 = nn.Conv1d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)

        self.enc4_1 = ResBlock(base_channels * 4, base_channels * 8, t_dim, num_groups, dropout)
        self.enc4_2 = ResBlock(base_channels * 8, base_channels * 8, t_dim, num_groups, dropout)
        self.enc4_attn = AttentionBlock(base_channels * 8, num_groups=num_groups)
        self.down4 = nn.Conv1d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1)

        self.bot_res1 = ResBlock(base_channels * 8, base_channels * 8, t_dim, num_groups, dropout)
        self.bot_attn = AttentionBlock(base_channels * 8, num_groups=num_groups)
        self.bot_res2 = ResBlock(base_channels * 8, base_channels * 8, t_dim, num_groups, dropout)

        self.up4 = nn.ConvTranspose1d(base_channels * 8, base_channels * 8, 4, stride=2, padding=1)
        self.dec4_1 = ResBlock(base_channels * 8 + base_channels * 8, base_channels * 4, t_dim, num_groups, dropout)
        self.dec4_2 = ResBlock(base_channels * 4, base_channels * 4, t_dim, num_groups, dropout)
        self.dec4_attn = AttentionBlock(base_channels * 4, num_groups=num_groups)

        self.up3 = nn.ConvTranspose1d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)
        self.dec3_1 = ResBlock(base_channels * 4 + base_channels * 4, base_channels * 2, t_dim, num_groups, dropout)
        self.dec3_2 = ResBlock(base_channels * 2, base_channels * 2, t_dim, num_groups, dropout)

        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.dec2_1 = ResBlock(base_channels * 2 + base_channels * 2, base_channels, t_dim, num_groups, dropout)
        self.dec2_2 = ResBlock(base_channels, base_channels, t_dim, num_groups, dropout)

        self.up1 = nn.ConvTranspose1d(base_channels, base_channels, 4, stride=2, padding=1)
        self.dec1_1 = ResBlock(base_channels + base_channels, base_channels, t_dim, num_groups, dropout)
        self.dec1_2 = ResBlock(base_channels, base_channels, t_dim, num_groups, dropout)

        self.out_proj = nn.Sequential(
            nn.GroupNorm(num_groups, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, 1, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.t_embedding(t)
        h = self.in_proj(x)

        s1 = self.enc1_2(self.enc1_1(h, t_emb), t_emb)
        d1 = self.down1(s1)

        s2 = self.enc2_2(self.enc2_1(d1, t_emb), t_emb)
        d2 = self.down2(s2)

        s3 = self.enc3_2(self.enc3_1(d2, t_emb), t_emb)
        d3 = self.down3(s3)

        s4 = self.enc4_attn(self.enc4_2(self.enc4_1(d3, t_emb), t_emb))
        d4 = self.down4(s4)

        b = self.bot_res2(self.bot_attn(self.bot_res1(d4, t_emb)), t_emb)

        u4 = self.up4(b)
        if u4.shape[-1] != s4.shape[-1]:
            u4 = F.interpolate(u4, size=s4.shape[-1], mode='nearest')
        d4_out = self.dec4_attn(self.dec4_2(self.dec4_1(torch.cat([u4, s4], dim=1), t_emb), t_emb))

        u3 = self.up3(d4_out)
        if u3.shape[-1] != s3.shape[-1]:
            u3 = F.interpolate(u3, size=s3.shape[-1], mode='nearest')
        d3_out = self.dec3_2(self.dec3_1(torch.cat([u3, s3], dim=1), t_emb), t_emb)

        u2 = self.up2(d3_out)
        if u2.shape[-1] != s2.shape[-1]:
            u2 = F.interpolate(u2, size=s2.shape[-1], mode='nearest')
        d2_out = self.dec2_2(self.dec2_1(torch.cat([u2, s2], dim=1), t_emb), t_emb)

        u1 = self.up1(d2_out)
        if u1.shape[-1] != s1.shape[-1]:
            u1 = F.interpolate(u1, size=s1.shape[-1], mode='nearest')
        d1_out = self.dec1_2(self.dec1_1(torch.cat([u1, s1], dim=1), t_emb), t_emb)

        return self.out_proj(d1_out)

class SpectralNoiseSampler(nn.Module):
    def __init__(self, sqrt_h):
        super().__init__()
        self.register_buffer('sqrt_h', sqrt_h)

    def set_h(self, sqrt_h):
        if isinstance(sqrt_h, np.ndarray):
            sqrt_h = torch.tensor(sqrt_h, dtype=torch.float32)
        self.sqrt_h.copy_(sqrt_h.to(self.sqrt_h.device))

    def color_noise(self, eps):
        L = eps.shape[-1]
        return torch.fft.irfft(self.sqrt_h * torch.fft.rfft(eps, n=L), n=L)

    def sample_xt(self, x0, x1, sigma_t, sigma_bar_t):
        s2 = sigma_t ** 2
        sb2 = sigma_bar_t ** 2
        denom = s2 + sb2 + 1e-8
        mu = (sb2 * x0 + s2 * x1) / denom
        std = torch.sqrt(torch.clamp(s2 * sb2 / denom, min=1e-8))
        return mu + std * self.color_noise(torch.randn_like(x0))

class NoiseSchedule:
    def __init__(self, sigma_max=1.0, g_min=1e-6, g_max=1.3e-4, num_steps=1000, device='cpu'):
        t_np = np.linspace(0.0, 1.0, num_steps + 1)
        g_np = np.where(
            t_np <= 0.5,
            g_min + (g_max - g_min) * (t_np / 0.5),
            g_max - (g_max - g_min) * ((t_np - 0.5) / 0.5),
        )
        sigma2_np = np.zeros_like(t_np)
        for i in range(1, len(t_np)):
            sigma2_np[i] = sigma2_np[i-1] + 0.5 * (g_np[i-1] + g_np[i]) * (t_np[i] - t_np[i-1])
        sigma2_np = sigma2_np / (sigma2_np[-1] + 1e-8) * (sigma_max ** 2)
        self.t = torch.tensor(t_np, dtype=torch.float32, device=device)
        self.sigma2 = torch.tensor(sigma2_np, dtype=torch.float32, device=device)

    def get_sigma(self, t):
        t_clamped = t.clamp(0.0, 1.0)
        idx = t_clamped * (len(self.t) - 1)
        idx_lo = idx.long().clamp(0, len(self.t) - 2)
        frac = idx - idx_lo.float()
        s2 = self.sigma2[idx_lo] + frac * (self.sigma2[idx_lo + 1] - self.sigma2[idx_lo])
        sb2 = self.sigma2[-1] - s2
        return torch.sqrt(s2.clamp(min=1e-8)).view(-1, 1, 1), torch.sqrt(sb2.clamp(min=1e-8)).view(-1, 1, 1)

class SpectralSBUNet(nn.Module):
    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        c = cfg or Config()
        self.in_channels = kwargs.get("in_channels", c.in_channels)
        self.loss_type = kwargs.get("loss_type", c.loss_type)
        base_channels = kwargs.get("base_channels", c.base_channels)
        t_emb_dim = kwargs.get("t_emb_dim", c.t_emb_dim)
        num_groups = kwargs.get("num_groups", c.num_groups)
        dropout = kwargs.get("dropout", c.dropout)
        sqrt_h_path = kwargs.get("sqrt_h_path", c.sqrt_h_path)
        seg_len = kwargs.get("seg_len", c.seg_len)

        if sqrt_h_path == "white" or sqrt_h_path is None or not os.path.exists(sqrt_h_path):
            sqrt_h = torch.ones(seg_len // 2 + 1, dtype=torch.float32)
        else:
            h_data = np.load(sqrt_h_path).astype(np.float32)
            sqrt_h = torch.tensor(np.sqrt(h_data) if h_data.ndim == 1 and not sqrt_h_path.endswith("sqrt_h.npy") else h_data)

        self.spectral_sampler = SpectralNoiseSampler(sqrt_h)
        self.unet = UNet1D(
            in_channels=self.in_channels,
            base_channels=base_channels,
            t_emb_dim=t_emb_dim,
            num_groups=num_groups,
            dropout=dropout
        )

    def set_spectral_h(self, sqrt_h_or_path):
        if isinstance(sqrt_h_or_path, str):
            if sqrt_h_or_path == "white":
                sqrt_h = np.ones(self.spectral_sampler.sqrt_h.shape[0], dtype=np.float32)
            else:
                raw = np.load(sqrt_h_or_path).astype(np.float32)
                sqrt_h = np.sqrt(raw) if not sqrt_h_or_path.endswith("sqrt_h.npy") else raw
        else:
            sqrt_h = sqrt_h_or_path
        self.spectral_sampler.set_h(sqrt_h)

    def forward(self, x, x1=None, t=None):
        if self.in_channels == 2:
            x_in = torch.cat([x, x1 if x1 is not None else x], dim=1)
        else:
            x_in = x
        return self.unet(x_in, t)

    def sample_xt(self, x0, x1, t, schedule):
        sigma_t, sigma_bar_t = schedule.get_sigma(t)
        return self.spectral_sampler.sample_xt(x0, x1, sigma_t, sigma_bar_t), sigma_t, sigma_bar_t

    @torch.no_grad()
    def sample(self, x1, schedule, n_steps=1, eps=1e-4):
        B = x1.shape[0]
        device = x1.device
        if n_steps == 1:
            t = torch.full((B,), 1.0 - eps, device=device)
            out = self.forward(x1, x1, t)
            if self.loss_type == "score":
                sigma_t, _ = schedule.get_sigma(t)
                return x1 - sigma_t * out
            return out

        timesteps = torch.linspace(1.0 - eps, eps, n_steps, device=device)
        xn = x1.clone()

        for i, t_val in enumerate(timesteps):
            t_batch = t_val.expand(B)
            out = self.forward(xn, x1, t_batch)
            sigma_t, sigma_bar_t = schedule.get_sigma(t_batch)
            if self.loss_type == "score":
                x0_pred = xn - sigma_t * out
            else:
                x0_pred = out

            if i < n_steps - 1:
                t_next = timesteps[i + 1].expand(B)
                sigma_next, sigma_bar_next = schedule.get_sigma(t_next)
                denom = sigma_next**2 + sigma_bar_next**2 + 1e-8
                w_x0 = sigma_bar_next**2 / denom
                w_x1 = sigma_next**2 / denom
                mu = w_x0 * x0_pred + w_x1 * x1
                std = torch.sqrt(torch.clamp(sigma_next**2 * sigma_bar_next**2 / denom, min=1e-8))
                noise = self.spectral_sampler.color_noise(torch.randn_like(xn))
                xn = mu + std * noise
            else:
                xn = x0_pred
        return xn
