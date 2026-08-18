import pickle
import wfdb

def prepare(nstdb_path="dataset/mitnoise", output_path="data_prep/mitnoise.pkl"):
    root = nstdb_path.rstrip("/") + "/"
    bw_signals, bw_fields = wfdb.rdsamp(root + "bw")
    em_signals, em_fields = wfdb.rdsamp(root + "em")
    ma_signals, ma_fields = wfdb.rdsamp(root + "ma")

    for key in bw_fields:
        print(key, bw_fields[key])
    for key in em_fields:
        print(key, em_fields[key])
    for key in ma_fields:
        print(key, ma_fields[key])

    with open(output_path, "wb") as f:
        pickle.dump([bw_signals, em_signals, ma_signals], f)

    print("=========================================================")
    print("MIT BIH data noise stress test database (NSTDB) saved as pickle")
    return [bw_signals, em_signals, ma_signals]