"""
    EEG Real-Time Recording and Visualization with Muse (Interaxon)

    Author: Fred Simard, RE-AK Technologies, 2025
    Website: https://www.re-ak.com
    Discord: https://discord.gg/XzeDHJf6ne

    Description:
    ------------
    This script acquires real-time EEG data from the Muse headset, applies
    ASR (Artifact Subspace Reconstruction) using the functions in ASR.py,
    and visualizes both the raw and cleaned EEG side-by-side.

    ASR pipeline (uses original ASR.py functions):
      - Accumulates the first CAL_SECONDS of live EEG as a calibration buffer.
      - Bandpass-filters it with bp_fir_zero() from ASR.py.
      - Fits an ASR model using asrpy (same library used in ASR.py).
      - Transforms every subsequent chunk with asr.transform().

    Core Features:
    --------------
    - Real-time EEG data acquisition using MuseProxy
    - Raw data saved to CSV
    - Side-by-side visualization: left = Raw EEG, right = ASR Cleaned EEG
    - Graceful handling of Ctrl+C to ensure proper shutdown

    Dependencies:
    -------------
    - Tested on Python 3.12
    - pyqtgraph, numpy, scipy, mne, asrpy
    - MuseProxy module (custom)
    - ASR.py (provides bp_fir_zero, ensure_finite_raw, ASR_CUTOFF, L_FREQ, H_FREQ)
"""


import csv
import os
import atexit
import json
import numpy as np
from time import sleep
import signal
from multiprocessing import Process, Queue, Event
from datetime import datetime
from pathlib import Path

import mne
from asrpy.asr import ASR

# ── Import helper functions and settings from your original ASR.py ─────────────
from ASR import (bp_fir_zero, ensure_finite_raw, load_raw_any,
                 ASR_CUTOFF, L_FREQ, H_FREQ, JSON_SFREQ)

from proxies.MuseProxy import MuseProxy
from utils.visualization import visualizer

MUSE_MAC_ADDRESS = "00:55:DA:B8:22:49"  # replace with your own MAC address

# ── Calibration mode ───────────────────────────────────────────────────────────
#   USE_FILE_CAL = False  →  live: use first CAL_SECONDS of the live EEG stream
#   USE_FILE_CAL = True   →  file: load CAL_FILE and fit ASR at startup
#                             (right panel cleans from the very first sample)
# ──────────────────────────────────────────────────────────────────────────────
USE_FILE_CAL = True                                  # use offline calibration file at startup
CAL_FILE     = "data/55_AWE_STIMULUS_MUSE.json"    # requested calibration file
PROMPT_CALIBRATION_AT_START = True
CAL_LIBRARY_SUBDIR = os.path.join("data", "clean_calibrations")
CAL_SELECTION_FILE = os.path.join(CAL_LIBRARY_SUBDIR, "selected_calibration.txt")

# ── Calibration settings (live mode only) ─────────────────────────────────────
CAL_SECONDS  = 120                            # seconds of live EEG used to calibrate
CAL_SAMPLES  = int(JSON_SFREQ * CAL_SECONDS) # = 5120 samples at 256 Hz

# ── Real-time epoch size (matches the offline ASR epoch length)
RT_EPOCH_SECONDS = 4.0
EPOCH_SAMPLES    = int(JSON_SFREQ * RT_EPOCH_SECONDS)
CH_NAMES         = ["TP9", "AF7", "AF8", "TP10"]

# ── ASR state ──────────────────────────────────────────────────────────────────
_cal_buffer   = []    # accumulates raw chunks during live calibration
_epoch_buffer = []    # accumulates raw chunks until we have 1 full second
_epoch_ts_buffer = []  # timestamps aligned with _epoch_buffer
_asr_model    = None  # asrpy ASR instance, set once calibration is done
_asr_fitted   = False
_mne_info     = mne.create_info(CH_NAMES, sfreq=JSON_SFREQ, ch_types="eeg")
_save_live_calibration = False


def _fit_asr_from_file(csv_path):
    """
    Fit ASR from a pre-recorded file using the exact offline ASR.py path:
      load_raw_any -> bp_fir_zero -> ASR.fit
    Returns True on success, False if file is missing/invalid (falls back to live).
    """
    global _asr_model, _asr_fitted, _mne_info

    path = os.path.join(script_dir, csv_path) if not os.path.isabs(csv_path) else csv_path

    if not os.path.exists(path):
        print(f"[ASR] CAL_FILE not found: '{path}'. Falling back to live calibration.")
        return False

    try:
        cal_raw = load_raw_any(Path(path))
    except Exception as e:
        print(f"[ASR] Failed to load CAL_FILE '{path}' ({type(e).__name__}: {e}). Falling back to live calibration.")
        return False

    if cal_raw.n_times < int(JSON_SFREQ):   # need at least 1 second
        print(f"[ASR] CAL_FILE too short ({cal_raw.n_times} samples). Falling back to live calibration.")
        return False

    print(f"[ASR] Loading calibration file: '{os.path.basename(path)}' ({cal_raw.n_times} samples)")

    # Keep the transform info aligned with calibration channels from offline loader.
    _mne_info = cal_raw.info.copy()

    # Identical to offline method: filter full calibration signal before fit.
    cal_bp = bp_fir_zero(cal_raw, band=(L_FREQ, H_FREQ))

    _asr_model = ASR(sfreq=JSON_SFREQ, cutoff=ASR_CUTOFF)
    _asr_model.fit(cal_bp)
    _asr_fitted = True

    print(f"[ASR] Model fitted from file. cutoff={ASR_CUTOFF}, band=({L_FREQ}–{H_FREQ} Hz)")
    print("[ASR] Real-time cleaning is now active from the first sample.")
    return True


"""
CSV file handling
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, "data"), exist_ok=True)
RAW_SESSION_DIR = os.path.join(script_dir, "data", "raw_sessions")
CLEAN_SESSION_DIR = os.path.join(script_dir, "data", "clean_sessions")
os.makedirs(RAW_SESSION_DIR, exist_ok=True)
os.makedirs(CLEAN_SESSION_DIR, exist_ok=True)


def _cal_library_dir() -> str:
    return os.path.join(script_dir, CAL_LIBRARY_SUBDIR)


def _cal_selection_file() -> str:
    return os.path.join(script_dir, CAL_SELECTION_FILE)


def _abs_from_script(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(script_dir, path)


def _rel_to_script(path: str) -> str:
    try:
        return os.path.relpath(path, script_dir)
    except ValueError:
        return path


def _list_calibration_files():
    candidates = []
    seen = set()
    search_dirs = [os.path.join(script_dir, "data"), _cal_library_dir()]

    for folder in search_dirs:
        if not os.path.isdir(folder):
            continue

        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(".json"):
                continue

            full_path = os.path.join(folder, name)
            if full_path in seen:
                continue

            seen.add(full_path)
            candidates.append(full_path)

    return candidates


def _load_persisted_calibration_path():
    sel_file = _cal_selection_file()
    if not os.path.exists(sel_file):
        return None

    try:
        with open(sel_file, "r", encoding="utf-8") as f:
            saved = f.read().strip()
        if not saved:
            return None
        full_path = _abs_from_script(saved)
        return full_path if os.path.exists(full_path) else None
    except Exception:
        return None


def _persist_selected_calibration(path: str):
    os.makedirs(_cal_library_dir(), exist_ok=True)
    with open(_cal_selection_file(), "w", encoding="utf-8") as f:
        f.write(_rel_to_script(path))


def _save_clean_calibration_json(samples_uV: np.ndarray):
    """Save a clean baseline recording as a Muse-style JSON file for future reuse."""
    if samples_uV.ndim != 2 or samples_uV.shape[1] != 4:
        return None

    os.makedirs(_cal_library_dir(), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(_cal_library_dir(), f"clean_cal_{ts}.json")

    payload = {
        "RAW_TP9": samples_uV[:, 0].astype(float).tolist(),
        "RAW_AF7": samples_uV[:, 1].astype(float).tolist(),
        "RAW_AF8": samples_uV[:, 2].astype(float).tolist(),
        "RAW_TP10": samples_uV[:, 3].astype(float).tolist(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    _persist_selected_calibration(out_path)
    print(f"[ASR] Saved clean calibration recording: '{out_path}'")
    return out_path


def _configure_calibration_workflow():
    """
    Choose calibration source at startup:
      1) Use an existing clean recording
      2) Record a new clean recording from live data and save it for reuse
    """
    global USE_FILE_CAL, CAL_FILE, _save_live_calibration

    if not PROMPT_CALIBRATION_AT_START:
        return

    default_from_setting = _abs_from_script(CAL_FILE)
    persisted = _load_persisted_calibration_path()
    default_file = persisted if persisted else (default_from_setting if os.path.exists(default_from_setting) else None)

    print("\n[ASR] Calibration setup")
    print("  1) Use existing clean recording")
    print("  2) Record new clean recording now (first live calibration window)")

    try:
        mode = input("Select mode [1/2] (default 1): ").strip()
    except EOFError:
        mode = "1"

    if mode == "2":
        USE_FILE_CAL = False
        _save_live_calibration = True
        print("[ASR] New clean recording will be captured from live data and saved for future runs.")
        return

    candidates = _list_calibration_files()
    if not candidates:
        print("[ASR] No calibration JSON files found. Switching to live capture mode.")
        USE_FILE_CAL = False
        _save_live_calibration = True
        return

    print("\n[ASR] Available calibration recordings:")
    default_index = None
    for idx, path in enumerate(candidates, start=1):
        marker = ""
        if default_file and os.path.normcase(path) == os.path.normcase(default_file):
            marker = " (default)"
            default_index = idx
        print(f"  {idx}) {_rel_to_script(path)}{marker}")

    try:
        raw_choice = input("Select calibration file number (Enter for default): ").strip()
    except EOFError:
        raw_choice = ""

    chosen_path = None
    if raw_choice:
        try:
            pick = int(raw_choice)
            if 1 <= pick <= len(candidates):
                chosen_path = candidates[pick - 1]
        except ValueError:
            chosen_path = None

    if chosen_path is None:
        if default_file and os.path.exists(default_file):
            chosen_path = default_file
        elif default_index is not None:
            chosen_path = candidates[default_index - 1]
        else:
            chosen_path = candidates[0]

    USE_FILE_CAL = True
    _save_live_calibration = False
    CAL_FILE = _rel_to_script(chosen_path)
    _persist_selected_calibration(chosen_path)
    print(f"[ASR] Using calibration recording: '{CAL_FILE}'")

now = datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

raw_filename = f"raw_session_{timestamp_str}.csv"
clean_filename = f"clean_session_{timestamp_str}.csv"

raw_file   = open(os.path.join(RAW_SESSION_DIR, raw_filename), "w", newline="")
raw_writer = csv.writer(raw_file)
raw_writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10"])

clean_file   = open(os.path.join(CLEAN_SESSION_DIR, clean_filename), "w", newline="")
clean_writer = csv.writer(clean_file)
clean_writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10"])

def close_file():
    raw_file.close()
    clean_file.close()

atexit.register(close_file)


# ── Queues for inter-process communication ─────────────────────────────────────
q         = Queue()   # raw EEG samples  →  left  visualizer panel
q_clean   = Queue()   # ASR-cleaned EEG  →  right visualizer panel
hsi_q     = Queue()   # HSI contact quality
battery_q = Queue()   # Battery level


"""
EEG callback

Sends raw data to the left visualizer panel.
During calibration: accumulates data and fits ASR using ASR.py's bp_fir_zero.
After calibration: transforms each chunk with asr.transform() and sends to right panel.
"""
def eeg_callback(timestamps, data):
    """
    Called when new EEG samples arrive from the Muse.
    data shape: (12, 4) in µV.
    """
    global _cal_buffer, _epoch_buffer, _epoch_ts_buffer, _asr_model, _asr_fitted, CAL_FILE

    # Raw EEG → left panel
    q.put(data)

    if not _asr_fitted:
        # ── Calibration phase: collect data ───────────────────────────────
        _cal_buffer.append(data.copy())
        total = sum(d.shape[0] for d in _cal_buffer)

        if total >= CAL_SAMPLES:
            print(f"[ASR] Calibration buffer full ({total} samples). Fitting ASR...")

            live_cal_uV = np.vstack(_cal_buffer)

            cal_V   = live_cal_uV.T * 1e-6   # (4, n_samples) in Volts
            cal_raw = mne.io.RawArray(cal_V, _mne_info.copy(), verbose=False)
            cal_raw = ensure_finite_raw(cal_raw)         # ASR.py: replace NaNs
            cal_bp  = bp_fir_zero(cal_raw,               # ASR.py: bandpass filter
                                  band=(L_FREQ, H_FREQ))

            _asr_model = ASR(sfreq=JSON_SFREQ, cutoff=ASR_CUTOFF)
            _asr_model.fit(cal_bp)

            if _save_live_calibration:
                saved_path = _save_clean_calibration_json(live_cal_uV)
                if saved_path:
                    CAL_FILE = _rel_to_script(saved_path)
                    print(f"[ASR] New clean recording selected for next runs: '{CAL_FILE}'")

            _asr_fitted = True
            _cal_buffer = []  # free memory
            print(f"[ASR] Model fitted. cutoff={ASR_CUTOFF}, band=({L_FREQ}–{H_FREQ} Hz)")
            print("[ASR] Real-time cleaning is now active.")

    else:
        # ── Cleaning phase: accumulate 1-second epoch, then clean ─────────
        # Mirror the offline pipeline as closely as possible:
        #   ensure_finite_raw(epoch) → bp_fir_zero(epoch) → asr.transform(epoch)
        _epoch_buffer.append(data.copy())
        _epoch_ts_buffer.append(np.asarray(timestamps).copy())
        epoch_total = sum(d.shape[0] for d in _epoch_buffer)

        if epoch_total >= EPOCH_SAMPLES:
            epoch_data    = np.vstack(_epoch_buffer)[:EPOCH_SAMPLES]  # (25, 4) µV
            epoch_ts      = np.concatenate(_epoch_ts_buffer)[:EPOCH_SAMPLES]
            _epoch_buffer = []   # reset for next epoch
            _epoch_ts_buffer = []

            try:
                epoch_V   = epoch_data.T * 1e-6
                epoch_raw = mne.io.RawArray(epoch_V, _mne_info.copy(), verbose=False)
                epoch_raw = ensure_finite_raw(epoch_raw)
                epoch_bp  = bp_fir_zero(epoch_raw, band=(L_FREQ, H_FREQ))
                cleaned_raw = _asr_model.transform(epoch_bp)

                if isinstance(cleaned_raw, mne.io.BaseRaw):
                    cleaned_uV = cleaned_raw.get_data().T * 1e6   # (25, 4) µV
                else:
                    cleaned_uV = np.asarray(cleaned_raw, float).T * 1e6

                for i in range(epoch_ts.shape[0]):
                    clean_writer.writerow([epoch_ts[i]] + cleaned_uV[i, :].tolist())

                q_clean.put(cleaned_uV)

            except Exception as e:
                print(f"[ASR] Transform error (forwarding raw): {e}")
                for i in range(epoch_ts.shape[0]):
                    clean_writer.writerow([epoch_ts[i]] + epoch_data[i, :].tolist())
                q_clean.put(epoch_data)

    # Write raw samples to CSV
    for i in range(data.shape[0]):
        raw_writer.writerow([timestamps[i]] + data[i, :].tolist())


def hsi_callback(hsi_values):
    """
    Called when a new HSI (Headband Signal Indicator) update arrives.
    hsi_values: list of 4 ints [TP9, AF7, AF8, TP10] — 1=good, 2=medium, 4=bad
    """
    hsi_q.put(hsi_values)


def battery_callback(pct: float):
    """
    Called when a new telemetry packet arrives with the battery percentage.
    """
    battery_q.put(pct)


# ------------------------------
# Graceful Shutdown Handling
# ------------------------------

shutdown_event = Event()

def signal_handler(sig, frame):
    """Handles Ctrl+C (SIGINT) to trigger a clean shutdown."""
    print("\n[INFO] Ctrl+C received. Initiating shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)


# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":

    _configure_calibration_workflow()

    # ── File-based calibration (like offline CAL_FILE) ────────────────────────
    if USE_FILE_CAL:
        _fit_asr_from_file(CAL_FILE)   # falls back to live mode if file is bad

    # Start visualization in a separate process
    vis_process = Process(target=visualizer,
                          args=(q, shutdown_event, hsi_q, battery_q, q_clean))
    vis_process.start()

    try:
        # Initialize Muse connection
        proxy = MuseProxy(MUSE_MAC_ADDRESS, eeg_callback,
                          telemetry_callback=battery_callback,
                          hsi_callback=hsi_callback)
        proxy.waitForConnected()

        # Optional buffer period to stabilize signal
        sleep(1)
        print("Initial padding, to stabilize signals...")
        sleep(20)

        # Start of live demo sequence
        print("Starting experience")

        # Phase 1: Stream EEG for 6 minutes (adjust to your demo needs)
        sleep(360)

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt caught in try-block.")

    finally:
        print("[MAIN] Cleaning up...")
        shutdown_event.set()

        # proxy.disconnect() can raise RuntimeError if MuseProxy's own
        # signal handler already closed the event loop — swallow it cleanly.
        try:
            proxy.disconnect()
        except Exception as e:
            print(f"[MAIN] proxy.disconnect() skipped ({type(e).__name__}: {e})")

        q.put(None)
        q_clean.put(None)
        vis_process.terminate()
        vis_process.join()
        print("[MAIN] Shutdown complete. Goodbye!")