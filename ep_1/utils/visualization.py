"""
Real-Time EEG Visualizer for Muse Headset (Lesson 1)

Author: Fred Simard, RE-AK Technologies, 2025
Website: https://www.re-ak.com
Discord: https://discord.gg/XzeDHJf6ne

Description:
------------
This module implements a real-time EEG visualizer using PyQtGraph and multiprocessing.

It receives EEG samples from a queue and plots the four Muse channels (TP9, AF7, AF8, TP10)
in separate subplots. A status panel at the top shows:
  - HSI (Headband Signal Indicator): coloured circles per electrode
      Green = good (1), Orange = medium (2), Red = bad (4)
  - Battery level: a compact bar gauge (0-100 %)

The window is split into two halves:
  - Left  : Raw EEG (from `queue`)
  - Right : ASR-Cleaned EEG (from `clean_queue`)

Main Components:
----------------
- `visualizer(queue, shutdown_event, hsi_queue=None, battery_queue=None, clean_queue=None)`:
    Launches a PyQtGraph-based GUI window in a separate process. Handles real-time EEG data
    visualization and graceful shutdown triggered via multiprocessing event.
"""

import numpy as np
from multiprocessing import Queue, Event
from utils.filters import apply_bandpass_filter

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, QtCore


def _qsize_policy(policy_name):
    """Return QSizePolicy enum value compatible with both PyQt5 and PyQt6."""
    if hasattr(QtWidgets.QSizePolicy, policy_name):
        return getattr(QtWidgets.QSizePolicy, policy_name)
    return getattr(QtWidgets.QSizePolicy.Policy, policy_name)


def _qt_enum(name, scoped_name):
    """Return Qt enum value compatible with both PyQt5 and PyQt6."""
    if hasattr(QtCore.Qt, name):
        return getattr(QtCore.Qt, name)
    scope = getattr(QtCore.Qt, scoped_name)
    return getattr(scope, name)


def _qfont_bold_weight():
    """Return QFont bold weight compatible with both PyQt5 and PyQt6."""
    if hasattr(QtGui.QFont, "Bold"):
        return QtGui.QFont.Bold
    return QtGui.QFont.Weight.Bold


def _qpainter_antialiasing_hint():
    """Return QPainter antialiasing hint compatible with both PyQt5 and PyQt6."""
    if hasattr(QtGui.QPainter, "Antialiasing"):
        return QtGui.QPainter.Antialiasing
    return QtGui.QPainter.RenderHint.Antialiasing


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _hsi_color(val):
    """Return an RGB tuple for an HSI value.

    None = unknown/no data yet  -> grey
    1    = good contact         -> green
    2    = medium contact       -> amber
    4    = bad contact          -> red
    """
    if val is None:
        return (100, 100, 115)   # grey  – waiting for data
    elif val == 1:
        return (80, 220, 100)    # green – good
    elif val == 2:
        return (255, 180, 40)    # amber – medium
    else:
        return (220, 60, 60)     # red   – bad


def _battery_color(pct: float):
    """Return an RGB tuple that transitions red→amber→green with battery %."""
    if pct >= 60:
        return (80, 220, 100)
    elif pct >= 25:
        return (255, 180, 40)
    else:
        return (220, 60, 60)


# ── Status panel widget ────────────────────────────────────────────────────────

class StatusPanel(QtWidgets.QWidget):
    """
    A compact Qt widget showing:
      - Four HSI circles (one per EEG electrode)
      - A horizontal battery bar
    """

    CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hsi   = [None, None, None, None]  # None = grey until first update
        self._batt  = None                       # None = unknown

        self.setMinimumHeight(70)
        self.setMaximumHeight(90)
        self.setSizePolicy(
            _qsize_policy("Expanding"),
            _qsize_policy("Fixed"),
        )

    def set_hsi(self, values):
        """Update HSI values (list of 4 ints)."""
        if len(values) == 4:
            self._hsi = list(values)
            self.update()

    def set_battery(self, pct: float):
        """Update battery percentage (0-100)."""
        self._batt = max(0.0, min(100.0, float(pct)))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(_qpainter_antialiasing_hint())

        try:
            w = self.width()
            h = self.height()

            # Dark background
            painter.fillRect(0, 0, w, h, QtGui.QColor(30, 30, 40))

            # ── HSI circles ───────────────────────────────────────────────────────
            circle_r  = 16
            spacing   = 90
            hsi_x0    = 20
            label_y   = h // 2 - circle_r - 4
            circle_y  = h // 2 - circle_r + 10

            font = QtGui.QFont("Segoe UI", 8, _qfont_bold_weight())
            painter.setFont(font)

            for i, (name, val) in enumerate(zip(self.CHANNEL_NAMES, self._hsi)):
                cx = hsi_x0 + i * spacing + circle_r
                r, g, b = _hsi_color(val)

                # Glow ring (skip for unknown state)
                if val is not None:
                    glow_pen = QtGui.QPen(QtGui.QColor(r, g, b, 80), 4)
                    painter.setPen(glow_pen)
                    painter.setBrush(_qt_enum("NoBrush", "BrushStyle"))
                    painter.drawEllipse(cx - circle_r - 3, circle_y - 3,
                                        (circle_r + 3) * 2, (circle_r + 3) * 2)

                # Filled circle
                painter.setPen(_qt_enum("NoPen", "PenStyle"))
                painter.setBrush(QtGui.QColor(r, g, b))
                painter.drawEllipse(cx - circle_r, circle_y,
                                    circle_r * 2, circle_r * 2)

                # Channel label
                painter.setPen(QtGui.QColor(200, 200, 210))
                painter.drawText(
                    QtCore.QRect(cx - spacing // 2, label_y, spacing, 18),
                    _qt_enum("AlignCenter", "AlignmentFlag"),
                    name,
                )

            # ── Battery bar ───────────────────────────────────────────────────────
            bar_x     = hsi_x0 + 4 * spacing + 10
            bar_w_max = w - bar_x - 20
            bar_h     = 22
            bar_y     = (h - bar_h) // 2

            # Border
            painter.setPen(QtGui.QPen(QtGui.QColor(120, 120, 140), 1))
            painter.setBrush(QtGui.QColor(50, 50, 65))
            painter.drawRoundedRect(bar_x, bar_y, bar_w_max, bar_h, 4, 4)

            if self._batt is not None:
                fill_w = int(bar_w_max * self._batt / 100.0)
                r, g, b = _battery_color(self._batt)
                painter.setPen(_qt_enum("NoPen", "PenStyle"))
                painter.setBrush(QtGui.QColor(r, g, b))
                painter.drawRoundedRect(bar_x + 1, bar_y + 1,
                                        max(0, fill_w - 2), bar_h - 2, 3, 3)

                # Battery % text
                painter.setPen(QtGui.QColor(230, 230, 240))
                font2 = QtGui.QFont("Segoe UI", 9, _qfont_bold_weight())
                painter.setFont(font2)
                painter.drawText(
                    QtCore.QRect(bar_x, bar_y, bar_w_max, bar_h),
                    _qt_enum("AlignCenter", "AlignmentFlag"),
                    f"{self._batt:.0f} %",
                )

                # Small battery icon to the left of bar
                painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 200), 1))
                painter.setBrush(_qt_enum("NoBrush", "BrushStyle"))
                icon_x = bar_x - 26
                icon_y = bar_y + 3
                painter.drawRect(icon_x, icon_y, 18, 16)
                painter.drawRect(icon_x + 18, icon_y + 4, 4, 8)   # nub
            else:
                # No data yet
                painter.setPen(QtGui.QColor(120, 120, 140))
                painter.setFont(QtGui.QFont("Segoe UI", 8))
                painter.drawText(
                    QtCore.QRect(bar_x, bar_y, bar_w_max, bar_h),
                    _qt_enum("AlignCenter", "AlignmentFlag"),
                    "Battery: --",
                )
        finally:
            painter.end()


# ── Main visualizer ────────────────────────────────────────────────────────────

def visualizer(queue: Queue, shutdown_event: Event,
               hsi_queue: Queue = None, battery_queue: Queue = None,
               clean_queue: Queue = None):
    """
    Real-time EEG plotter using PyQtGraph.

    The window is split into two halves:
      - Left  panel : Raw EEG (4 channels stacked)
      - Right panel : ASR-Cleaned EEG (4 channels stacked)

    Parameters:
    -----------
    queue : multiprocessing.Queue
        Queue through which raw EEG samples (shape: [samples, 4]) are received.

    shutdown_event : multiprocessing.Event
        Event used to signal that the visualizer should terminate cleanly.

    hsi_queue : multiprocessing.Queue, optional
        Queue carrying HSI lists [TP9, AF7, AF8, TP10] (ints 1/2/4).

    battery_queue : multiprocessing.Queue, optional
        Queue carrying battery percentage floats (0-100).

    clean_queue : multiprocessing.Queue, optional
        Queue through which ASR-cleaned EEG samples (shape: [samples, 4]) are received.
        When provided, the right panel is populated with cleaned data.
    """

    CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
    # Raw  : cooler blue-ish tones
    RAW_COLORS = [
        (100, 180, 255),   # blue  – TP9
        (100, 255, 150),   # green – AF7
        (255, 180,  80),   # amber – AF8
        (255, 100, 100),   # red   – TP10
    ]
    # Clean : warmer / slightly brighter variants
    CLEAN_COLORS = [
        ( 60, 220, 255),   # cyan  – TP9
        ( 60, 255, 120),   # lime  – AF7
        (255, 220,  60),   # gold  – AF8
        (255,  60, 160),   # pink  – TP10
    ]

    app = QtWidgets.QApplication([])

    # ── Main window ───────────────────────────────────────────────────────────
    main_win = QtWidgets.QMainWindow()
    main_win.setWindowTitle("Real-time EEG – Muse  |  Left: Raw   Right: ASR Cleaned")
    main_win.resize(1600, 860)

    central = QtWidgets.QWidget()
    v_layout = QtWidgets.QVBoxLayout(central)
    v_layout.setContentsMargins(6, 6, 6, 6)
    v_layout.setSpacing(4)

    # Status panel (HSI + battery) – spans full width
    status_panel = StatusPanel()
    v_layout.addWidget(status_panel)

    # ── Column header labels ──────────────────────────────────────────────────
    header_row = QtWidgets.QHBoxLayout()

    def _make_header(text, color):
        lbl = QtWidgets.QLabel(text)
        lbl.setAlignment(_qt_enum("AlignCenter", "AlignmentFlag"))
        lbl.setStyleSheet(
            f"color: {color}; font-size: 15px; font-weight: bold;"
            "background: transparent; padding: 2px;"
        )
        return lbl

    header_row.addWidget(_make_header("Raw EEG",         "#7bbfff"))
    header_row.addWidget(_make_header("ASR Cleaned EEG", "#7bffda"))
    v_layout.addLayout(header_row)

    # ── Side-by-side PyQtGraph widgets ───────────────────────────────────────
    h_layout = QtWidgets.QHBoxLayout()

    pg_raw   = pg.GraphicsLayoutWidget()
    pg_clean = pg.GraphicsLayoutWidget()
    pg_raw.setBackground((20, 22, 30))
    pg_clean.setBackground((22, 25, 30))

    h_layout.addWidget(pg_raw)
    h_layout.addWidget(pg_clean)
    v_layout.addLayout(h_layout)

    main_win.setCentralWidget(central)
    main_win.show()

    # ── Build subplots for both panels ────────────────────────────────────────
    buffer_size = 256 * 3  # 3 seconds at 256 Hz

    raw_curves   = []
    clean_curves = []
    raw_buffers   = [np.zeros(buffer_size) for _ in range(4)]
    clean_buffers = [np.zeros(buffer_size) for _ in range(4)]

    for i, name in enumerate(CHANNEL_NAMES):
        is_last = (i == len(CHANNEL_NAMES) - 1)

        # ── Raw panel ────────────────────────────────────────────────────
        pr = pg_raw.addPlot(
            title=f"<span style='color:#aaa'>{name}</span>"
        )
        pr.showGrid(x=True, y=True, alpha=0.25)
        pr.setLabel('left', 'µV', **{'color': '#888', 'font-size': '8pt'})
        pr.setLabel('bottom',
                    'Samples (3 s rolling)' if is_last else '',
                    **{'color': '#888', 'font-size': '8pt'})
        pr.setYRange(-200, 200, padding=0.05)
        cr = pr.plot(pen=pg.mkPen(color=RAW_COLORS[i], width=1.5))
        raw_curves.append(cr)
        pg_raw.nextRow()

        # ── Clean panel ──────────────────────────────────────────────────
        pc = pg_clean.addPlot(
            title=f"<span style='color:#aaa'>{name}</span>"
        )
        pc.showGrid(x=True, y=True, alpha=0.25)
        pc.setLabel('left', 'µV', **{'color': '#888', 'font-size': '8pt'})
        pc.setLabel('bottom',
                    'Samples (3 s rolling)' if is_last else '',
                    **{'color': '#888', 'font-size': '8pt'})
        pc.setYRange(-200, 200, padding=0.05)
        cc = pc.plot(pen=pg.mkPen(color=CLEAN_COLORS[i], width=1.5))
        clean_curves.append(cc)
        pg_clean.nextRow()

    # ── Update callback (called every 30 ms) ──────────────────────────────────
    def update():
        # --- Raw EEG data ---
        while not queue.empty():
            samples = queue.get()
            if samples is None:
                break
            for i in range(samples.shape[0]):
                sample = samples[i, :]
                for j in range(4):
                    raw_buffers[j] = np.roll(raw_buffers[j], -1)
                    raw_buffers[j][-1] = sample[j]
                    raw_curves[j].setData(raw_buffers[j])

        # --- ASR-Cleaned EEG data ---
        if clean_queue is not None:
            while not clean_queue.empty():
                samples_clean = clean_queue.get()
                if samples_clean is None:
                    break
                for i in range(samples_clean.shape[0]):
                    sample_c = samples_clean[i, :]
                    for j in range(4):
                        clean_buffers[j] = np.roll(clean_buffers[j], -1)
                        clean_buffers[j][-1] = sample_c[j]
                        clean_curves[j].setData(clean_buffers[j])

        # --- HSI updates ---
        if hsi_queue is not None:
            while not hsi_queue.empty():
                hsi_vals = hsi_queue.get()
                status_panel.set_hsi(hsi_vals)

        # --- Battery updates ---
        if battery_queue is not None:
            while not battery_queue.empty():
                batt = battery_queue.get()
                status_panel.set_battery(batt)

        if shutdown_event.is_set():
            print("[VISUALIZER] Shutdown signal received. Closing app...")
            app.quit()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(30)

    if hasattr(app, "exec_"):
        app.exec_()
    else:
        app.exec()
