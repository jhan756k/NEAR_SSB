import glob
import math
import pickle
import sys
from pathlib import Path
import numpy as np
import wfdb
from scipy.signal import resample_poly

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.preprocessing import preprocess_signal

def prepare(qt_path="dataset/qtdb", output_path="data_prep/qtdb.pkl"):
    new_fs = 360
    record_paths = glob.glob(qt_path.rstrip("/") + "/*.dat")
    qtdb_signals = {}

    for path in record_paths:
        base = path[:-4]
        record_name = base.replace("\\", "/").split("/")[-1]
        signal, fields = wfdb.rdsamp(base)
        ann = wfdb.rdann(base, "pu1")

        ann_type = np.array(ann.symbol)
        ann_samples = ann.sample
        p_idx = ann_samples[ann_type == "p"]
        s_idx = ann_samples[ann_type == "("]
        r_idx = ann_samples[ann_type == "N"]

        ind = np.zeros(len(p_idx), dtype=np.int64)
        for i in range(len(p_idx)):
            arr = np.where(p_idx[i] > s_idx)[0]
            ind[i] = arr[-1]

        p_start = s_idx[ind] - int(0.04 * fields["fs"])
        aux_sig = signal[:, 0]

        beats = []
        for i in range(len(p_start) - 1):
            remove = (r_idx > p_start[i]) & (r_idx < p_start[i + 1])
            if np.sum(remove) < 2:
                beats.append(aux_sig[p_start[i]:p_start[i + 1]])

        beats_re = []
        for beat in beats:
            l = math.ceil(len(beat) * new_fs / fields["fs"])
            norm_beat = list(reversed(beat)) + list(beat) + list(reversed(beat))
            res = resample_poly(norm_beat, new_fs, fields["fs"])
            processed_beat = np.asarray(res[l - 1:2 * l - 1], dtype=np.float32)
            processed_beat = preprocess_signal(processed_beat, new_fs)
            beats_re.append(processed_beat)

        qtdb_signals[record_name] = beats_re

    with open(output_path, "wb") as f:
        pickle.dump(qtdb_signals, f)

    print("=========================================================")
    print("MIT QT database saved as pickle file")
    return qtdb_signals