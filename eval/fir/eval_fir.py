import argparse
from pathlib import Path
import numpy as np
from scipy import signal
import torch

from datahandle_fir import ensure_preprocessed_data, load_pickles, build_test_split
from metrics import SSD, MAD, PRD, COS_SIM, SNR, SNR_improvement


def fir_filter(x_test: np.ndarray, fs: float = 360):
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 40 / nyquist
    numtaps = 101 #filter order +1

    b = signal.firwin(numtaps, [low, high], pass_zero=False)
    x_squeezed = np.squeeze(x_test, axis=-1)

    filtered_signals = signal.filtfilt(b, 1.0, x_squeezed, axis=1)
    return np.expand_dims(filtered_signals, axis=2)


def evaluate_fir(x_test: np.ndarray, y_test: np.ndarray):
    y_pred = fir_filter(x_test)

    clean_np = np.squeeze(y_test, axis=-1)
    noisy_np = np.squeeze(x_test, axis=-1)
    out_np = np.squeeze(y_pred, axis=-1)

    return {
        "SSD": SSD(clean_np, out_np),
        "MAD": MAD(clean_np, out_np),
        "PRD": PRD(clean_np, out_np),
        "COS_SIM": COS_SIM(y_test, y_pred),
        "SNR_in": SNR(clean_np, noisy_np),
        "SNR_out": SNR(clean_np, out_np),
        "SNR_improvement": SNR_improvement(noisy_np, out_np, clean_np)
    }


def summarize_metrics(name: str, values: np.ndarray):
    print(f"{name}: {values.mean():.6f} +/- {values.std():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FIR filter")
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

    all_results = []
    all_levels = []

    for noise_type in args.noise_types:
        print(f"Evaluating noise type {noise_type}")

        x_test, y_test, rnd_test = build_test_split(qtdb, nstdb, noise_version=noise_type, seed=args.seed)
        result = evaluate_fir(x_test, y_test)

        all_results.append(result)
        all_levels.append(rnd_test)

    merged = {k: np.concatenate([r[k] for r in all_results], axis=0) for k in all_results[0]}
    n_level = np.concatenate(all_levels, axis=0)

    print("\n" + "=" * 60)
    print(f"FIR evaluation from preprocessed data")
    for metric in ["SSD", "MAD", "PRD", "COS_SIM", "SNR_in", "SNR_out", "SNR_improvement"]:
        summarize_metrics(metric, merged[metric])

    segs = [0.2, 0.6, 1.0, 1.5, 2.0]
    for lo, hi in zip(segs[:-1], segs[1:]):
            idx = np.argwhere(np.logical_and(n_level >= lo, n_level <= hi)).reshape(-1)
            if idx.size == 0:
                continue
            print("-" * 60)
            print(f"{lo} <= noise <= {hi} (n={idx.size})")
            for metric in ["SSD", "MAD", "PRD", "COS_SIM", "SNR_in", "SNR_out", "SNR_improvement"]:
                summarize_metrics(metric, merged[metric][idx])


if __name__ == "__main__":
    main()