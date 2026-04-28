# 🧠 MuseExperiment – EEG Visual Demos with Muse by Interaxon

**Author:** Fred Simard, [RE-AK Technologies](https://www.re-ak.com), 2025  
📣 [Join our Discord](https://discord.gg/XzeDHJf6ne) for support, discussion, and updates!

---

This repository provides a minimal experimental setup to get started with the **Muse** EEG headband by **Interaxon**. It's designed to accompany a YouTube tutorial series, focusing on teaching the basics of BCI hacking.

It then provides code samples for the various explanations and demonstrations presented in the Youtube Videos.

---

## 📦 Contents

- ✅ **`ep_1/`** – A beginner-friendly example that:
  - Connects to a Muse headset
  - Starts a real-time visualizer using PyQtGraph
  - Saves filtered EEG data to CSV
  - Demonstrates common EEG artifacts: blinks, jaw clenches, frowning, noise
  - Youtube video: https://youtu.be/eTBOwD8-0VM

- ✅ **`ep_2/`** – Capturing the Alpha Wave. Example that:
  - Connects to a Muse headset
  - Implements the described experiment paradigm
  - Saves filtered EEG data to CSV
  - Visualize the results using: Spectrogram and Power Spectrum Density
  - Youtube video: https://youtu.be/H6OsqBXr_7A

- ✅ **`ep_3/`** – Biofeedback Experience Design and Unsupervised Calibration, using GMM:
  - Use the sample dataset recorded in episode 2 (refer to episode 2 to record your own data)
  - Describes the parameters of the biofeedback experiment we are developing
  - Presents two different models: Z-Score transform and two-states GMM
  - Visualize the results
  - Youtube video: https://youtu.be/HBL3W3tV23E

- ✅ **`ep_3/`** – Biofeedback Experience Design and Unsupervised Calibration, using GMM:
  - Use the sample dataset recorded in episode 2 (refer to episode 2 to record your own data)
  - Describes the parameters of the biofeedback experiment we are developing
  - Presents two different models: Z-Score transform and two-states GMM
  - Visualize the results
  - Youtube video: https://youtu.be/HBL3W3tV23E


---

## 🚀 Getting Started (episode 1)

### Requirements

Make sure you have the following installed:

- edit `experimentManager.py` to set your Muse Mac address

```bash
pip install -r ep_1/requirements.txt
```

## 🔍 Find Your Muse MAC Address

```bash
cd ep_1
python ble_scanner.py
```

## ▶️ Run the Live Visualizer

```bash
cd ep_1
python lesson_1.py
```

This will:

Connect to your Muse headset

Display real-time EEG plots

Save data to:
ep_1/data/my_file_{timestamp}.csv

### Running on Linux
It should work, but you need to remove the soft_beep method as it's specific for Windows.


## 🛠️ Status – July 27, 2025
Just published Lesson 2. It is functional and tested with the Muse 2 on Windows. Contributions and issues are welcome.

## 👤 Author
Fred Simard
RE-AK Technologies
📅 2025
🌐 www.re-ak.com
💬 [Join our Discord](https://discord.gg/XzeDHJf6ne)

