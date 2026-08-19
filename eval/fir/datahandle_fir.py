import pickle
from pathlib import Path
import numpy as np

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
