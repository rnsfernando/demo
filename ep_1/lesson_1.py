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
import numpy as np
from time import sleep
import signal
from multiprocessing import Process, Queue, Event
from datetime import datetime

import mne
from asrpy.asr import ASR

# ── Import helper functions and settings from your original ASR.py ─────────────
from ASR import (bp_fir_zero, ensure_finite_raw,
                 ASR_CUTOFF, L_FREQ, H_FREQ, JSON_SFREQ)

from proxies.MuseProxy import MuseProxy
from utils.visualization import visualizer

MUSE_MAC_ADDRESS = "00:55:DA:B8:22:49"  # replace with your own MAC address

# ── Calibration mode ───────────────────────────────────────────────────────────
#   USE_FILE_CAL = False  →  live: use first CAL_SECONDS of the live EEG stream
#   USE_FILE_CAL = True   →  file: load CAL_FILE CSV and fit ASR at startup
#                             (right panel cleans from the very first sample)
# ──────────────────────────────────────────────────────────────────────────────
USE_FILE_CAL = False                          # ← flip this to switch modes
CAL_FILE     = "data/my_clean_session.csv"    # ← used only when USE_FILE_CAL = True

# ── Calibration settings (live mode only) ─────────────────────────────────────
CAL_SECONDS  = 20                            # seconds of live EEG used to calibrate
CAL_SAMPLES  = int(JSON_SFREQ * CAL_SECONDS) # = 5120 samples at 256 Hz

# ── Real-time epoch size (shorter = lower delay, less accurate filter)
# Note: bp_fir_zero needs 500+ samples so it is only used during calibration.
# For the transform step, raw 25-sample chunks are passed directly to ASR.
RT_EPOCH_SECONDS = 5                              # ← change this to adjust
EPOCH_SAMPLES    = int(JSON_SFREQ * RT_EPOCH_SECONDS)  # = 25 samples @ 256 Hz
CH_NAMES         = ["TP9", "AF7", "AF8", "TP10"]

# ── ASR state ──────────────────────────────────────────────────────────────────
_cal_buffer   = []    # accumulates raw chunks during live calibration
_epoch_buffer = []    # accumulates raw chunks until we have 1 full second
_asr_model    = None  # asrpy ASR instance, set once calibration is done
_asr_fitted   = False
_mne_info     = mne.create_info(CH_NAMES, sfreq=JSON_SFREQ, ch_types="eeg")


def _fit_asr_from_file(csv_path):
    """
    Fit ASR from a pre-recorded CSV file — mirrors the offline CAL_FILE approach.
    CSV must have columns: Timestamp, TP9, AF7, AF8, TP10  (saved by lesson_1.py).
    Returns True on success, False if file is missing/invalid (falls back to live).
    """
    global _asr_model, _asr_fitted
    import csv as _csv

    path = os.path.join(script_dir, csv_path) if not os.path.isabs(csv_path) else csv_path

    if not os.path.exists(path):
        print(f"[ASR] CAL_FILE not found: '{path}'. Falling back to live calibration.")
        return False

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                rows.append([float(row["TP9"]),  float(row["AF7"]),
                             float(row["AF8"]),  float(row["TP10"])])
            except (KeyError, ValueError):
                continue

    if len(rows) < int(JSON_SFREQ):   # need at least 1 second
        print(f"[ASR] CAL_FILE too short ({len(rows)} samples). Falling back to live calibration.")
        return False

    print(f"[ASR] Loading calibration file: '{os.path.basename(path)}' ({len(rows)} samples)")

    cal_V   = np.array(rows, dtype=float).T * 1e-6   # (4, n_samples) V
    cal_raw = mne.io.RawArray(cal_V, _mne_info.copy(), verbose=False)
    cal_raw = ensure_finite_raw(cal_raw)              # ASR.py: replace NaNs
    cal_bp  = bp_fir_zero(cal_raw, band=(L_FREQ, H_FREQ))  # ASR.py: bandpass

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

now = datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"my_file_{timestamp_str}.csv"

eeg_file   = open(os.path.join(script_dir, "data", filename), "w", newline="")
eeg_writer = csv.writer(eeg_file)
eeg_writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10"])

def close_file():
    eeg_file.close()

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
    global _cal_buffer, _epoch_buffer, _asr_model, _asr_fitted

    # Raw EEG → left panel
    q.put(data)

    if not _asr_fitted:
        # ── Calibration phase: collect data ───────────────────────────────
        _cal_buffer.append(data.copy())
        total = sum(d.shape[0] for d in _cal_buffer)

        if total >= CAL_SAMPLES:
            print(f"[ASR] Calibration buffer full ({total} samples). Fitting ASR...")

            cal_V   = np.vstack(_cal_buffer).T * 1e-6   # (4, n_samples) in Volts
            cal_raw = mne.io.RawArray(cal_V, _mne_info.copy(), verbose=False)
            cal_raw = ensure_finite_raw(cal_raw)         # ASR.py: replace NaNs
            cal_bp  = bp_fir_zero(cal_raw,               # ASR.py: bandpass filter
                                  band=(L_FREQ, H_FREQ))

            _asr_model = ASR(sfreq=JSON_SFREQ, cutoff=ASR_CUTOFF)
            _asr_model.fit(cal_bp)

            _asr_fitted = True
            _cal_buffer = []  # free memory
            print(f"[ASR] Model fitted. cutoff={ASR_CUTOFF}, band=({L_FREQ}–{H_FREQ} Hz)")
            print("[ASR] Real-time cleaning is now active.")

    else:
        # ── Cleaning phase: accumulate 1-second epoch, then clean ─────────
        # This mirrors the offline pipeline exactly:
        #   bp_fir_zero(epoch) → asr.transform(epoch)
        _epoch_buffer.append(data.copy())
        epoch_total = sum(d.shape[0] for d in _epoch_buffer)

        if epoch_total >= EPOCH_SAMPLES:
            epoch_data    = np.vstack(_epoch_buffer)[:EPOCH_SAMPLES]  # (25, 4) µV
            _epoch_buffer = []   # reset for next epoch

            try:
                # Pass directly to ASR — bp_fir_zero skipped because FIR
                # filters need 500+ samples; calibration was already filtered.
                epoch_V   = epoch_data.T * 1e-6
                epoch_raw = mne.io.RawArray(epoch_V, _mne_info.copy(), verbose=False)
                cleaned_raw = _asr_model.transform(epoch_raw)

                if isinstance(cleaned_raw, mne.io.BaseRaw):
                    cleaned_uV = cleaned_raw.get_data().T * 1e6   # (25, 4) µV
                else:
                    cleaned_uV = np.asarray(cleaned_raw, float).T * 1e6

                q_clean.put(cleaned_uV)

            except Exception as e:
                print(f"[ASR] Transform error (forwarding raw): {e}")
                q_clean.put(epoch_data)

    # Write raw samples to CSV
    for i in range(data.shape[0]):
        eeg_writer.writerow([timestamps[i]] + data[i, :].tolist())


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