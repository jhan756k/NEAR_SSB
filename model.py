import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_res_blocks=2, use_attention=False, num_groups=8, dropout=0.0):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(in_channels if i == 0 else out_channels, out_channels, t_emb_dim, num_groups, dropout)
            for i in range(num_res_blocks)
        ])
        self.attn = AttentionBlock(out_channels, num_groups=num_groups) if use_attention else None
        self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        for res in self.res_blocks:
            x = res(x, t_emb)
        if self.attn is not None:
            x = self.attn(x)
        return x, self.downsample(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, t_emb_dim, num_res_blocks=2, use_attention=False, num_groups=8, dropout=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, 4, stride=2, padding=1)
        self.res_blocks = nn.ModuleList([
            ResBlock(in_channels + skip_channels if i == 0 else out_channels, out_channels, t_emb_dim, num_groups, dropout)
            for i in range(num_res_blocks)
        ])
        self.attn = AttentionBlock(out_channels, num_groups=num_groups) if use_attention else None

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        for res in self.res_blocks:
            x = res(x, t_emb)
        if self.attn is not None:
            x = self.attn(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, channels, t_emb_dim, num_groups=8, dropout=0.0):
        super().__init__()
        self.res1 = ResBlock(channels, channels, t_emb_dim, num_groups, dropout)
        self.attn = AttentionBlock(channels, num_groups=num_groups)
        self.res2 = ResBlock(channels, channels, t_emb_dim, num_groups, dropout)

    def forward(self, x, t_emb):
        return self.res2(self.attn(self.res1(x, t_emb)), t_emb)

class SpectralNoiseSampler(nn.Module):
    def __init__(self, sqrt_h):
        super().__init__()
        self.register_buffer('sqrt_h', sqrt_h)

    def color_noise(self, eps):
        L = eps.shape[-1]
        return torch.fft.irfft(self.sqrt_h * torch.fft.rfft(eps, n=L), n=L)

    def sample_xt(self, x0, x1, sigma_t, sigma_bar_t):
        s2 = sigma_t ** 2
        sb2 = sigma_bar_t ** 2
        mu = (sb2 * x0 + s2 * x1) / (s2 + sb2)
        std = torch.sqrt(s2 * sb2 / (s2 + sb2))
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
        sigma2_np = sigma2_np / sigma2_np[-1] * sigma_max ** 2
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
    def __init__(self, seg_len=512, base_channels=64, channel_mults=(1, 2, 4, 8), num_res_blocks=2, t_emb_dim=128, num_groups=8, dropout=0.0, sqrt_h_path="data_prep/spectral_h.npy"):
        super().__init__()
        self.seg_len = seg_len
        sqrt_h = torch.tensor(np.sqrt(np.load(sqrt_h_path).astype(np.float32)))
        self.spectral_sampler = SpectralNoiseSampler(sqrt_h)
        self.t_embedding = TimestepEmbedding(t_emb_dim)
        t_emb_full = t_emb_dim * 4
        self.input_proj = nn.Conv1d(1, base_channels, 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        in_ch = base_channels
        skip_channels = []
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.encoder_blocks.append(EncoderBlock(
                in_ch, out_ch, t_emb_full, num_res_blocks,
                use_attention=(i == len(channel_mults) - 1),
                num_groups=num_groups, dropout=dropout,
            ))
            skip_channels.append(out_ch)
            in_ch = out_ch

        self.bottleneck = Bottleneck(in_ch, t_emb_full, num_groups, dropout)

        self.decoder_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            skip_ch = skip_channels[-(i + 1)]
            out_ch = base_channels * (channel_mults[len(channel_mults) - 2 - i] if i < len(channel_mults) - 1 else 1)
            self.decoder_blocks.append(DecoderBlock(
                in_ch, skip_ch, out_ch, t_emb_full, num_res_blocks,
                use_attention=(i == 0),
                num_groups=num_groups, dropout=dropout,
            ))
            in_ch = out_ch

        self.output_proj = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, 1, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.t_embedding(t)
        x = self.input_proj(x)
        skips = []
        for enc in self.encoder_blocks:
            skip, x = enc(x, t_emb)
            skips.append(skip)
        x = self.bottleneck(x, t_emb)
        for dec in self.decoder_blocks:
            x = dec(x, skips.pop(), t_emb)
        return self.output_proj(x)

    def sample_xt(self, x0, x1, t, schedule):
        sigma_t, sigma_bar_t = schedule.get_sigma(t)
        return self.spectral_sampler.sample_xt(x0, x1, sigma_t, sigma_bar_t), sigma_t, sigma_bar_t

    @torch.no_grad()
    def sample(self, x1, schedule, n_steps=50, eps=1e-4):
        B = x1.shape[0]
        device = x1.device
        timesteps = torch.linspace(1.0 - eps, eps, n_steps, device=device)
        xn = x1.clone()

        for i, t_val in enumerate(timesteps):
            t_batch = t_val.expand(B)
            '''
            score = self.forward(xn, t_batch)
            sigma_t, _ = schedule.get_sigma(t_batch)
            x0_pred = xn - sigma_t * score
            '''
            x0_pred = self.forward(xn, t_batch).clamp(-1.0, 1.0)
            

            if i < n_steps - 1:
                t_next = timesteps[i + 1].expand(B)
                sigma_next, sigma_bar_next = schedule.get_sigma(t_next)
                
                w_x0 = sigma_bar_next**2 / (sigma_next**2 + sigma_bar_next**2)
                w_x1 = sigma_next**2 / (sigma_next**2 + sigma_bar_next**2)
                mu = w_x0 * x0_pred + w_x1 * x1

                std = torch.sqrt(sigma_next**2 * sigma_bar_next**2 / (sigma_next**2 + sigma_bar_next**2))

                noise = self.spectral_sampler.color_noise(torch.randn_like(xn))
                xn = mu + std * noise
                
            else:
                xn = x0_pred
                
        return xn