from pathlib import Path
from typing import Union, List, Tuple
import json
import shutil

import numpy as np
import mne
from mne.filter import filter_data
from asrpy.asr import ASR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from IPython.display import FileLink, display, HTML
except ImportError:
    FileLink = display = HTML = None  # not running inside Jupyter


# ============================================================
# INPUT FILES
# ============================================================

CAL_FILE  = "/kaggle/input/emognition/55/55_AWE_STIMULUS_MUSE.json"
CAL_WINDOW = None

DATA_DIR   = "/kaggle/input/emognition"
PATTERN    = "*STIMULUS_MUSE.json"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTDIR = Path("/kaggle/working/Output_ASR_AllEpochs_First2Subjects")


# ============================================================
# SAMPLING & FILTERING SETTINGS
# ============================================================

JSON_SFREQ = 256.0
L_FREQ     = 1.0
H_FREQ     = 40.0


# ============================================================
# EPOCHING
# ============================================================

EPOCH_LENGTH  = 2.0
EPOCH_OVERLAP = 0.0


# ============================================================
# ASR
# ============================================================

ASR_CUTOFF = 24.0


# ============================================================
# PLOTTING SETTINGS
# ============================================================

PLOT_WINDOW   = (0, 40.0)
PLOT_CHANNELS = "RAW_TP9"

FIG_W   = 50
FIG_H   = 8
FIG_DPI = 300

# ============================================================
# BASIC UTILITIES
# ============================================================

def ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_finite_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Replace NaNs and inf values with channel median.
    If a full channel is bad, replace with 0.
    """
    X = raw.get_data()

    if np.isfinite(X).all():
        return raw

    X = X.copy()

    for ch in range(X.shape[0]):
        bad = ~np.isfinite(X[ch])

        if bad.any():
            good = ~bad
            fill = np.median(X[ch][good]) if good.any() else 0.0
            X[ch][bad] = fill

    return mne.io.RawArray(X, raw.info.copy(), verbose=False)


# ============================================================
# JSON / EDF LOADING
# ============================================================

def _to_float_array(v) -> np.ndarray:
    if isinstance(v, list):
        out = []

        for x in v:
            if isinstance(x, list) and len(x) == 1:
                x = x[0]

            try:
                out.append(float(x))
            except Exception:
                out.append(np.nan)

        return np.asarray(out, float)

    try:
        return np.asarray([float(v)], float)
    except Exception:
        return np.asarray([np.nan], float)


def load_raw_json(path: Path, sfreq: float = JSON_SFREQ) -> mne.io.RawArray:
    """
    Load Muse-style JSON file.

    Expected channels:
        RAW_TP9
        RAW_AF7
        RAW_AF8
        RAW_TP10

    Input values are assumed to be in microvolts.
    MNE stores EEG data in volts.
    """
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    keys = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

    ch_names = []
    arrs = []

    for k in keys:
        if k in d:
            arrs.append(_to_float_array(d[k]))
            ch_names.append(k)

    if not arrs:
        for k, v in d.items():
            if k.startswith("RAW_"):
                arrs.append(_to_float_array(v))
                ch_names.append(k)

    if not arrs:
        raise RuntimeError(f"No RAW_* channels found in {path}")

    min_len = min(len(a) for a in arrs)

    data_uV = np.vstack([a[:min_len] for a in arrs])
    data_V  = data_uV * 1e-6

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(ch_names)
    )

    raw = mne.io.RawArray(data_V, info, verbose=False)
    raw = ensure_finite_raw(raw)

    print(f"[INFO] Loaded JSON: {path.name}")
    print(f"       Channels: {ch_names}")
    print(f"       Samples : {raw.n_times}")
    print(f"       Duration: {raw.n_times / sfreq:.2f} s")

    return raw


def load_raw_edf(path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=True)
    raw = ensure_finite_raw(raw)
    return raw


def load_raw_any(path: Path) -> mne.io.BaseRaw:
    if str(path).lower().endswith(".json"):
        return load_raw_json(path, sfreq=JSON_SFREQ)
    else:
        return load_raw_edf(path)


# ============================================================
# FILTERING
# ============================================================

def bp_fir_zero(
    raw: mne.io.BaseRaw,
    band: Tuple[float, float] = (L_FREQ, H_FREQ)
) -> mne.io.BaseRaw:
    """
    Apply zero-phase FIR band-pass filter to full signal.
    """
    sfreq = raw.info["sfreq"]
    X = raw.get_data()

    Xf = filter_data(
        X,
        sfreq=sfreq,
        l_freq=band[0],
        h_freq=band[1],
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )

    print(f"[INFO] FIR band-pass applied: {band[0]}–{band[1]} Hz")

    return mne.io.RawArray(Xf, raw.info.copy(), verbose=False)


# ============================================================
# EPOCHING
# ============================================================

def make_epochs(
    raw: mne.io.BaseRaw,
    length_sec: float,
    overlap_sec: float
) -> mne.Epochs:
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=length_sec,
        overlap=overlap_sec,
        preload=True,
        verbose="ERROR",
    )

    print(
        f"[INFO] Created {len(epochs)} epochs "
        f"({length_sec}s, overlap={overlap_sec}s)"
    )

    return epochs


# ============================================================
# ASR FOR EVERY EPOCH
# ============================================================

def apply_asr_on_all_epochs(
    X_ep: np.ndarray,
    epochs_info,
    raw_cal_bp: mne.io.BaseRaw,
    cutoff: float
) -> np.ndarray:
    """
    Apply ASR to every epoch.

    Input:
        X_ep shape = (n_epochs, n_channels, n_times)

    Output:
        X_clean shape = same as X_ep
    """
    n_epochs, n_ch, n_t = X_ep.shape

    asr = ASR(
        sfreq=raw_cal_bp.info["sfreq"],
        cutoff=cutoff
    )

    asr.fit(raw_cal_bp)

    print("[INFO] ASR fitted using calibration file")
    print(f"       ASR cutoff = {cutoff}")

    X_clean = X_ep.copy()

    for i in range(n_epochs):
        ep_raw = mne.io.RawArray(
            X_ep[i],
            epochs_info.copy(),
            verbose=False
        )

        ep_clean = asr.transform(ep_raw)

        if isinstance(ep_clean, mne.io.BaseRaw):
            ep_data = ep_clean.get_data()
        else:
            ep_data = np.asarray(ep_clean, float)

        L = min(ep_data.shape[1], n_t)
        X_clean[i, :, :L] = ep_data[:, :L]

    print("[INFO] ASR applied to every epoch")

    return X_clean


# ============================================================
# CONCATENATE EPOCHS BACK TO CONTINUOUS SIGNAL
# ============================================================

def build_continuous_from_epochs(
    X_ep: np.ndarray,
    base_info
) -> mne.io.BaseRaw:
    """
    Convert epoch array back to continuous RawArray.

    From:
        X_ep = (n_epochs, n_channels, n_times)

    To:
        X_cat = (n_channels, n_epochs * n_times)
    """
    n_epochs, n_ch, n_t = X_ep.shape

    X_cat = (
        X_ep
        .transpose(1, 0, 2)
        .reshape(n_ch, n_epochs * n_t)
    )

    raw_cat = mne.io.RawArray(
        X_cat,
        base_info.copy(),
        verbose=False
    )

    return raw_cat


# ============================================================
# CHANNEL SELECTION
# ============================================================

def pick_channels(
    raw: mne.io.BaseRaw,
    sel: Union[str, int, List[Union[str, int]]]
) -> List[int]:

    if isinstance(sel, str):
        return [raw.ch_names.index(sel)]

    if isinstance(sel, int):
        return [sel]

    idxs = []

    for s in sel:
        if isinstance(s, int):
            idxs.append(s)
        elif isinstance(s, str):
            idxs.append(raw.ch_names.index(s))

    return idxs


# ============================================================
# PLOTTING
# ============================================================

def plot_signal(
    raw: mne.io.BaseRaw,
    window: Tuple[float, float],
    ch_sel,
    title: str,
    fname: Path
):
    """
    Save one signal plot for the selected channel.
    """
    t0, t1 = window

    max_time = raw.n_times / raw.info["sfreq"]

    if t0 >= max_time:
        print(f"[WARN] Plot window starts after signal end. Skipping {fname}")
        return

    t1 = min(t1, max_time)

    seg = raw.copy().crop(
        t0,
        t1,
        include_tmax=False
    )

    t = seg.times + t0

    for idx in pick_channels(seg, ch_sel):
        y = seg.get_data()[idx] * 1e6

        plt.figure(figsize=(FIG_W, FIG_H), dpi=FIG_DPI)
        plt.plot(t, y, lw=0.8)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.title(f"{title} — {seg.ch_names[idx]}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

        print(f"[OK] Saved plot: {fname}")


# ============================================================
# SINGLE FILE PIPELINE
# ============================================================

def run_single_file(proc_file: Path, outdir: Path):
    """
    Processing order:
        1. Load raw EEG file
        2. Load ASR calibration file
        3. Band-pass calibration file
        4. Band-pass target file
        5. Epoch target file
        6. Apply ASR to every epoch
        7. Concatenate cleaned epochs
        8. Save raw, band-passed, and final cleaned plots
    """
    raw_proc = load_raw_any(proc_file)
    raw_cal  = load_raw_any(Path(CAL_FILE))

    # Filter calibration signal
    raw_cal_bp_full = bp_fir_zero(
        raw_cal,
        band=(L_FREQ, H_FREQ)
    )

    if CAL_WINDOW is None:
        raw_cal_bp = raw_cal_bp_full
    else:
        raw_cal_bp = raw_cal_bp_full.copy().crop(*CAL_WINDOW)

    # Filter processing signal
    raw_proc_bp = bp_fir_zero(
        raw_proc,
        band=(L_FREQ, H_FREQ)
    )

    # Epoch filtered signal
    epochs = make_epochs(
        raw_proc_bp,
        EPOCH_LENGTH,
        EPOCH_OVERLAP
    )

    X_ep = epochs.get_data()

    # ASR on every epoch
    X_clean_ep = apply_asr_on_all_epochs(
        X_ep,
        epochs_info=epochs.info,
        raw_cal_bp=raw_cal_bp,
        cutoff=ASR_CUTOFF
    )

    # Build continuous band-passed and cleaned signals
    raw_bp_cat = build_continuous_from_epochs(
        X_ep,
        epochs.info
    )

    raw_clean_cat = build_continuous_from_epochs(
        X_clean_ep,
        epochs.info
    )

    # Save only 3 plots
    plot_signal(
        raw_proc,
        PLOT_WINDOW,
        PLOT_CHANNELS,
        "Raw signal",
        outdir / "01_raw.png"
    )

    plot_signal(
        raw_bp_cat,
        PLOT_WINDOW,
        PLOT_CHANNELS,
        "Band-pass filtered signal",
        outdir / "02_bandpassed.png"
    )

    plot_signal(
        raw_clean_cat,
        PLOT_WINDOW,
        PLOT_CHANNELS,
        "Final ASR cleaned signal",
        outdir / "03_final_asr_cleaned.png"
    )

    return raw_proc, raw_bp_cat, raw_clean_cat


# ============================================================
# BATCH PROCESSING: FIRST 2 SUBJECTS ONLY
# ============================================================

def run_batch_process():
    data_dir = Path(DATA_DIR)

    all_files = sorted(data_dir.rglob(PATTERN))

    if not all_files:
        raise RuntimeError(f"No files found in {DATA_DIR}")

    # Subject ID is the part before the first underscore.
    # Only first 2 subjects are selected.
    subject_ids = sorted({p.stem.split("_")[0] for p in all_files})[:2]

    files = [
        p for p in all_files
        if p.stem.split("_")[0] in subject_ids
    ]

    print("\n===================================")
    print("[INFO] First 2 subjects selected")
    print("===================================")
    print(subject_ids)
    print(f"[INFO] Total files selected: {len(files)}")

    ensure_outdir(OUTDIR)

    for path in files:
        fname = path.stem
        participant = fname.split("_")[0]

        folder_name = f"{fname}_cleaned"
        outdir = OUTDIR / participant / folder_name
        outdir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print(f"[BATCH] Processing: {fname}")
        print(f"[BATCH] Subject   : {participant}")
        print("==============================")

        raw_proc, raw_bp_cat, raw_clean_cat = run_single_file(
            proc_file=path,
            outdir=outdir
        )

        # Save cleaned JSON in microvolts
        cleaned_json = outdir / f"{fname}_cleaned.json"

        data_uv = raw_clean_cat.get_data() * 1e6

        out = {}

        for ch_idx, ch_name in enumerate(raw_clean_cat.ch_names):
            out[ch_name] = data_uv[ch_idx].tolist()

        with cleaned_json.open("w", encoding="utf-8") as f:
            json.dump(out, f)

        print(f"[OK] Saved cleaned JSON: {cleaned_json}")

    print("\n[INFO] Batch processing completed.")


# ============================================================
# ZIP OUTPUT FOLDER
# ============================================================

def zip_output_folder():
    zip_base = str(OUTDIR)

    zip_path = shutil.make_archive(
        base_name=zip_base,
        format="zip",
        root_dir=OUTDIR.parent,
        base_dir=OUTDIR.name
    )

    print("\n[OK] Zipped output folder:")
    print(zip_path)

    return Path(zip_path)


# ============================================================
# DISPLAY DOWNLOAD LINK
# ============================================================

def show_download_link(zip_path: Path):
    if not zip_path.exists():
        print(f"[ERROR] Zip file not found: {zip_path}")
        return

    print(f"\n[OK] Zip file ready: {zip_path}")

    # Normal Kaggle/Jupyter clickable file link
    display(FileLink(str(zip_path)))

    # Extra HTML download button/link
    display(
        HTML(
            f"""
            <br>
            <a href="files/{zip_path.name}" download
               style="
                   background-color:#1976d2;
                   color:white;
                   padding:10px 16px;
                   text-decoration:none;
                   border-radius:6px;
                   font-weight:bold;
                   display:inline-block;
               ">
               Download ZIP
            </a>
            """
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_batch_process()
    zip_path = zip_output_folder()
    show_download_link(zip_path)
