import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from denoising_model_small import ConditionalModel
from main_model import DDPM


TEST_SET = {
    "sel123",
    "sel233",
    "sel302",
    "sel307",
    "sel820",
    "sel853",
    "sel16420",
    "sel16795",
    "sele0106",
    "sele0121",
    "sel32",
    "sel49",
    "sel14046",
    "sel15814",
}


def ensure_preprocessed_data(repo_root: Path, qtdb_path: Path, nstdb_path: Path, force: bool) -> tuple[Path, Path]:
    prep_dir = repo_root / "data_prep"
    prep_dir.mkdir(parents=True, exist_ok=True)

    qtdb_pkl = prep_dir / "qtdb.pkl"
    nstdb_pkl = prep_dir / "mitnoise.pkl"
    prep_qtdb_path = prep_dir / "prep_qtdb.py"
    prep_nstdb_path = prep_dir / "prep_nstdb.py"

    pickles_are_current = (
        qtdb_pkl.exists()
        and nstdb_pkl.exists()
        and qtdb_pkl.stat().st_mtime >= prep_qtdb_path.stat().st_mtime
        and nstdb_pkl.stat().st_mtime >= prep_nstdb_path.stat().st_mtime
    )

    if force or not pickles_are_current:
        import sys

        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from data_prep.prep_qtdb import prepare as prep_qtdb
        from data_prep.prep_nstdb import prepare as prep_nstdb

        print("Preparing QTDB/NSTDB pickles using data_prep scripts...")
        prep_qtdb(qt_path=str(qtdb_path), output_path=str(qtdb_pkl))
        prep_nstdb(nstdb_path=str(nstdb_path), output_path=str(nstdb_pkl))

    return qtdb_pkl, nstdb_pkl


def load_pickles(qtdb_pkl: Path, nstdb_pkl: Path):
    with qtdb_pkl.open("rb") as f:
        qtdb = pickle.load(f)
    with nstdb_pkl.open("rb") as f:
        nstdb = pickle.load(f)
    return qtdb, nstdb


def build_test_split(qtdb: dict, nstdb: list, noise_version: int, seed: int, samples: int = 512):
    rng = np.random.default_rng(seed)

    bw_signals = np.array(nstdb[0])
    bw_noise_channel1_a = bw_signals[0 : int(bw_signals.shape[0] / 2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0] / 2) : -1, 0]
    bw_noise_channel2_a = bw_signals[0 : int(bw_signals.shape[0] / 2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0] / 2) : -1, 1]

    if noise_version == 1:
        noise_test = bw_noise_channel2_b
    elif noise_version == 2:
        noise_test = bw_noise_channel1_b
    else:
        raise ValueError("noise_version must be 1 or 2")

    beats_test = []
    skipped = 0
    for signal_name, beats in qtdb.items():
        for beat in beats:
            if signal_name not in TEST_SET:
                continue
            b_sq = np.asarray(beat)
            b_np = np.zeros(samples, dtype=np.float32)
            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skipped += 1
                continue
            b_np[init_padding : b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2.0
            beats_test.append(b_np)

    if not beats_test:
        raise RuntimeError("No test beats found. Verify QTDB path and record names.")

    rnd_test = rng.integers(low=20, high=200, size=len(beats_test)) / 100.0

    sn_test = []
    noise_index = 0
    eps = 1e-8

    for i, beat in enumerate(beats_test):
        noise = noise_test[noise_index : noise_index + samples]
        beat_max_value = np.max(beat) - np.min(beat)
        noise_max_value = np.max(noise) - np.min(noise)

        if beat_max_value < eps:
            alpha = 0.0
        else:
            ase = noise_max_value / (beat_max_value + eps)
            alpha = rnd_test[i] / (ase + eps)

        signal_noise = beat + alpha * noise
        sn_test.append(signal_noise.astype(np.float32))

        noise_index += samples
        if noise_index > (len(noise_test) - samples):
            noise_index = 0

    x_test = np.expand_dims(np.array(sn_test, dtype=np.float32), axis=2)
    y_test = np.expand_dims(np.array(beats_test, dtype=np.float32), axis=2)

    print(f"Built test split for noise_type={noise_version}: {len(beats_test)} beats, skipped={skipped}")
    return x_test, y_test, rnd_test


def metric_ssd(y, y_pred):
    return np.sum(np.square(y - y_pred), axis=1)


def metric_mad(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1)


def metric_prd(y, y_pred):
    n = np.sum(np.square(y_pred - y), axis=1)
    d = np.sum(np.square(y_pred - np.mean(y, axis=1, keepdims=True)), axis=1)
    return np.sqrt(n / (d + 1e-12)) * 100


def metric_cos_sim(y, y_pred):
    y2 = np.squeeze(y, axis=-1)
    p2 = np.squeeze(y_pred, axis=-1)
    dot = np.sum(y2 * p2, axis=1)
    yn = np.linalg.norm(y2, axis=1)
    pn = np.linalg.norm(p2, axis=1)
    return (dot / (yn * pn + 1e-12)).reshape(-1, 1)


def metric_snr(y_true, y_est):
    n = np.sum(np.square(y_true), axis=1)
    d = np.sum(np.square(y_est - y_true), axis=1)
    return 10 * np.log10((n + 1e-12) / (d + 1e-12))


def evaluate_one_noise_type(model: DDPM, x_test: np.ndarray, y_test: np.ndarray, shots: int, batch_size: int, device: torch.device):
    x_t = torch.FloatTensor(x_test).permute(0, 2, 1)
    y_t = torch.FloatTensor(y_test).permute(0, 2, 1)

    test_set = TensorDataset(y_t, x_t)
    loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    ssd_total = []
    mad_total = []
    prd_total = []
    cos_total = []
    snr_noise = []
    snr_recon = []
    snr_improve = []

    with torch.no_grad():
        for clean_batch, noisy_batch in loader:
            clean_batch = clean_batch.to(device)
            noisy_batch = noisy_batch.to(device)

            if shots > 1:
                output = 0
                for _ in range(shots):
                    output += model.denoising(noisy_batch)
                output /= shots
            else:
                output = model.denoising(noisy_batch)

            clean_np = clean_batch.permute(0, 2, 1).cpu().numpy()
            noisy_np = noisy_batch.permute(0, 2, 1).cpu().numpy()
            out_np = output.permute(0, 2, 1).cpu().numpy()

            ssd_total.append(metric_ssd(clean_np, out_np))
            mad_total.append(metric_mad(clean_np, out_np))
            prd_total.append(metric_prd(clean_np, out_np))
            cos_total.append(metric_cos_sim(clean_np, out_np))
            snr_noise.append(metric_snr(clean_np, noisy_np))
            snr_recon.append(metric_snr(clean_np, out_np))
            snr_improve.append(metric_snr(clean_np, out_np) - metric_snr(clean_np, noisy_np))

    return {
        "ssd": np.concatenate(ssd_total, axis=0),
        "mad": np.concatenate(mad_total, axis=0),
        "prd": np.concatenate(prd_total, axis=0),
        "cos": np.concatenate(cos_total, axis=0),
        "snr_in": np.concatenate(snr_noise, axis=0),
        "snr_out": np.concatenate(snr_recon, axis=0),
        "snr_imp": np.concatenate(snr_improve, axis=0),
    }


def summarize_metrics(name: str, values: np.ndarray):
    print(f"{name}: {values.mean():.6f} +/- {values.std():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DESCOD using pickles produced by data_prep scripts")
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--qtdb-path", type=str, default="dataset/qtdb")
    parser.add_argument("--nstdb-path", type=str, default="dataset/mitnoise")
    parser.add_argument("--force-prep", action="store_true", help="Rebuild data_prep/qtdb.pkl and data_prep/mitnoise.pkl")
    parser.add_argument("--noise-types", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    qtdb_path = (repo_root / args.qtdb_path).resolve()
    nstdb_path = (repo_root / args.nstdb_path).resolve()

    if not qtdb_path.exists():
        raise FileNotFoundError(f"QTDB path not found: {qtdb_path}")
    if not nstdb_path.exists():
        raise FileNotFoundError(f"NSTDB path not found: {nstdb_path}")
    

    qtdb_pkl, nstdb_pkl = ensure_preprocessed_data(repo_root, qtdb_path, nstdb_path, force=args.force_prep)
    qtdb, nstdb = load_pickles(qtdb_pkl, nstdb_pkl)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "train": {"feats": 80},
        "diffusion": {
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": 50,
            "schedule": "quad",
        },
    }

    all_results = []
    all_levels = []

    for noise_type in args.noise_types:
        ckpt = Path(__file__).resolve().parent / "check_points" / f"noise_type_{noise_type}" / "model.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        base_model = ConditionalModel(config["train"]["feats"]).to(device)
        model = DDPM(base_model, config, str(device))
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        x_test, y_test, rnd_test = build_test_split(qtdb, nstdb, noise_version=noise_type, seed=args.seed)
        result = evaluate_one_noise_type(model, x_test, y_test, args.shots, args.batch_size, device)

        all_results.append(result)
        all_levels.append(rnd_test)

    merged = {k: np.concatenate([r[k] for r in all_results], axis=0) for k in all_results[0]}
    n_level = np.concatenate(all_levels, axis=0)

    print("=" * 60)
    print(f"DESCOD evaluation from preprocessed data ({args.shots}-shot)")
    summarize_metrics("ssd", merged["ssd"])
    summarize_metrics("mad", merged["mad"])
    summarize_metrics("prd", merged["prd"])
    summarize_metrics("cos_sim", merged["cos"])
    summarize_metrics("snr_in", merged["snr_in"])
    summarize_metrics("snr_out", merged["snr_out"])
    summarize_metrics("snr_improve", merged["snr_imp"])

    segs = [0.2, 0.6, 1.0, 1.5, 2.0]
    for lo, hi in zip(segs[:-1], segs[1:]):
        idx = np.argwhere(np.logical_and(n_level >= lo, n_level <= hi)).reshape(-1)
        if idx.size == 0:
            continue
        print("-" * 60)
        print(f"{lo} <= noise <= {hi} (n={idx.size})")
        summarize_metrics("ssd", merged["ssd"][idx])
        summarize_metrics("mad", merged["mad"][idx])
        summarize_metrics("prd", merged["prd"][idx])
        summarize_metrics("cos_sim", merged["cos"][idx])
        summarize_metrics("snr_in", merged["snr_in"][idx])
        summarize_metrics("snr_out", merged["snr_out"][idx])
        summarize_metrics("snr_improve", merged["snr_imp"][idx])


if __name__ == "__main__":
    main()
