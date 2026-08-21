import fit_psd_savgol
import visualize_psd

freqs, h_emp, h_smooth = fit_psd_savgol.fit(
    qtdb_path="data_prep/qtdb.pkl",
    output_path="data_prep/spectral_h_savgol.npy",
    fs=360,
    nperseg=256,
    noverlap=128,
    window=11,
    polyorder=3,
    anchor_freq=2.0,
    blend_width=3.0
)

visualize_psd.plot(
    freqs_path="data_prep/spectral_h_savgol_freqs.npy",
    emp_path="data_prep/spectral_h_savgol_emp.npy",
    smooth_path="data_prep/spectral_h_savgol.npy",
    output_path="data_prep/psd_fit.png"
)