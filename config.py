from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    seg_len: int = 512
    fs: int = 250
    data_dir: str = "data_prep"
    output_dir: str = "checkpoints"
    results_dir: str = "results"
    qtdb_pkl: str = "data_prep/bw_qtdb.pkl"
    nstdb_pkl: str = "data_prep/bw_mitnoise.pkl"
    reference_rnd_test: str = "data_prep/bw_rnd_test.npy"
    sqrt_h_path: str = "data_prep/bw_spectral_h.npy"
    loss_type: str = "x0"
    in_channels: int = 2
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    t_emb_dim: int = 128
    num_groups: int = 8
    dropout: float = 0.1
    sigma_max: float = 1.0
    g_min: float = 1e-6
    g_max: float = 1.3e-4
    n_schedule_steps: int = 1000
    eps: float = 1e-4
    batch_size: int = 128
    epochs: int = 400
    lr: float = 1e-3
    eta_min: float = 1e-6
    patience: int = 50
    save_every: int = 50
    n_inference_steps: int = 1
    device: str = "cuda"
