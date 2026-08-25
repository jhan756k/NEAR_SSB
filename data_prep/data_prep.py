import os
import pickle
import numpy as np
from data_prep import prep_nstdb, prep_qtdb

def prepare(
    noise_version=1,
    qt_path="dataset/qtdb",
    nstdb_path="dataset/mitnoise",
    output_dir="data_prep",
    force_prep=False,
    qtdb_pkl=None,
    nstdb_pkl=None
):
    print("Getting the Data ready ... ")
    np.random.seed(1234)
    os.makedirs(output_dir, exist_ok=True)

    qt_output = qtdb_pkl if qtdb_pkl is not None else os.path.join(output_dir, "bw_qtdb.pkl")
    noise_output = nstdb_pkl if nstdb_pkl is not None else os.path.join(output_dir, "bw_mitnoise.pkl")

    if qtdb_pkl is None and (force_prep or not os.path.exists(qt_output)):
        prep_qtdb.prepare(qt_path=qt_path, output_path=qt_output)
    else:
        print("Using cached file: " + qt_output)

    if nstdb_pkl is None and (force_prep or not os.path.exists(noise_output)):
        prep_nstdb.prepare(nstdb_path=nstdb_path, output_path=noise_output)
    else:
        print("Using cached file: " + noise_output)

    with open(qt_output, "rb") as f:
        qtdb = pickle.load(f)
    with open(noise_output, "rb") as f:
        nstdb = pickle.load(f)

    signals = np.array(nstdb[2]) #0: bw, 1: em, 2: ma
    noise_channel1_a = signals[0:int(signals.shape[0] / 2), 0]
    noise_channel1_b = signals[int(signals.shape[0] / 2):-1, 0]
    noise_channel2_a = signals[0:int(signals.shape[0] / 2), 1]
    noise_channel2_b = signals[int(signals.shape[0] / 2):-1, 1]

    if noise_version == 1:
        noise_test = noise_channel2_b
        noise_train = noise_channel1_a
    elif noise_version == 2:
        noise_test = noise_channel1_b
        noise_train = noise_channel2_a
    else:
        raise Exception("Sorry, noise_version should be 1 or 2")

    beats_train = []
    beats_test = []

    test_set = [
        "sel123", "sel233", "sel302", "sel307", "sel820", "sel853", "sel16420",
        "sel16795", "sele0106", "sele0121", "sel32", "sel49", "sel14046", "sel15814"
    ]

    samples = 512
    skip_beats = 0
    for signal_name in list(qtdb.keys()):
        for b in qtdb[signal_name]:
            b_np = np.zeros(samples)
            b_sq = np.array(b)
            init_padding = 16

            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue

            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)

    sn_train = []
    sn_test = []

    noise_index = 0
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    for i in range(len(beats_train)):
        noise = noise_train[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / ase
        sn_train.append(beats_train[i] + alpha * noise)
        noise_index += samples

        if noise_index > (len(noise_train) - samples):
            noise_index = 0

    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    np.save(output_dir.rstrip("/") + "/rnd_test.npy", rnd_test)
    print("rnd_test shape: " + str(rnd_test.shape))

    for i in range(len(beats_test)):
        noise = noise_test[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_test[i]) - np.min(beats_test[i])
        noise_max_value = np.max(noise) - np.min(noise)
        ase = noise_max_value / beat_max_value
        alpha = rnd_test[i] / ase
        sn_test.append(beats_test[i] + alpha * noise)
        noise_index += samples

        if noise_index > (len(noise_test) - samples):
            noise_index = 0

    x_train = np.expand_dims(np.array(sn_train), axis=2)
    y_train = np.expand_dims(np.array(beats_train), axis=2)
    x_test = np.expand_dims(np.array(sn_test), axis=2)
    y_test = np.expand_dims(np.array(beats_test), axis=2)

    print("Dataset ready to use.")
    return [x_train, y_train, x_test, y_test]

if __name__ == "__main__":
    prepare(noise_version=1, qt_path="dataset/qtdb", nstdb_path="dataset/mitnoise", output_dir="data_prep")