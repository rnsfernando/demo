"""
Muse BLE EEG Proxy for Real-Time Streaming and Signal Management

Author: Fred Simard, RE-AK Technologies, 2025
Website: https://www.re-ak.com
Discord: https://discord.gg/XzeDHJf6ne

Description:
------------
This module provides real-time Bluetooth Low Energy (BLE) streaming support for the Muse EEG headset 
(Interaxon). It handles connection, reconnection, signal decoding, and data delivery in an asynchronous 
and fault-tolerant way.

Key Class:
----------
MuseProxy:
    High-level wrapper to manage connection lifecycle, reconnection, signal handling, and clean shutdown.
    It embeds a BLE communication client (MuseBleak) and runs its loop in a separate background thread 
    using asyncio, allowing easy integration into scripts or UIs.

Usage:
------
MuseProxy handles EEG data streaming, including:
- BLE connection and reconnection handling
- Signal unpacking (bit-decoded)
- Per-sample timestamping
- Safe shutdown on SIGINT or system exit

This file is meant to be imported and used in larger EEG experiments or applications where live Muse data
needs to be accessed and processed.

Dependencies:
-------------
- bleak: for BLE communication
- bitstring, numpy: for unpacking and processing EEG packets
- asyncio, threading: for async architecture
"""


import threading
import csv
import atexit
import asyncio
import numpy as np
import bitstring
from bleak import BleakClient, BleakError
from time import time, sleep
import platform
import subprocess
import signal


MUSE_GATT_ATTRS = {
    "STREAM_TOGGLE": '273e0001-4c4d-454d-96be-f03bac821358',
    "TP9":           '273e0003-4c4d-454d-96be-f03bac821358',
    "AF7":           '273e0004-4c4d-454d-96be-f03bac821358',
    "AF8":           '273e0005-4c4d-454d-96be-f03bac821358',
    "TP10":          '273e0006-4c4d-454d-96be-f03bac821358',
    "TELEMETRY":     '273e000b-4c4d-454d-96be-f03bac821358',  # battery + temp
    "CONTROL":       '273e0001-4c4d-454d-96be-f03bac821358',  # HSI / control stream
}

SAMPLE_RATE = 256


class MuseProxy:
    """
    MuseProxy(mac_address, eeg_callback, telemetry_callback=None, hsi_callback=None)

    Main entry point for interacting with the Muse EEG headset.

    Responsibilities:
    -----------------
    - Wraps around MuseBleak (BLE implementation) to provide a high-level interface.
    - Starts an asyncio loop in a background thread to maintain BLE communication.
    - Registers signal and exit handlers for graceful shutdown on Ctrl+C or process exit.
    - Provides a simple blocking `waitForConnected()` method to allow downstream code to synchronize.

    Parameters:
    -----------
    mac_address : str
        The Bluetooth MAC address of the Muse device to connect to.

    eeg_callback : callable
        A callback function that receives EEG data. It must accept (timestamps, data) as arguments.
        - timestamps: 1D numpy array of floats
        - data: 2D numpy array with shape (channels, samples)

    telemetry_callback : callable, optional
        Called with a float battery percentage (0-100) whenever a telemetry packet arrives.

    hsi_callback : callable, optional
        Called with a list of 4 ints [TP9, AF7, AF8, TP10] where 1=good, 2=medium, 4=bad.

    Public Methods:
    ---------------
    waitForConnected():
        Blocks execution until a BLE connection is established.

    disconnect():
        Gracefully stops the BLE client and terminates the event loop.
    """
    
    
    
    def __init__(self, mac_address, eeg_callback,
                 telemetry_callback=None, hsi_callback=None):
        self.muse = MuseBleak(mac_address, eeg_callback,
                              telemetry_callback=telemetry_callback,
                              hsi_callback=hsi_callback)
        self.runner = AsyncRunnerThread(main, self.muse)

        def handle_exit(signum=None, frame=None):
            print(f"Exiting... signal={signum}")
            self.runner.stop()

        # Call on Ctrl+C or kill
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)

        # Call on normal interpreter exit
        atexit.register(handle_exit)

        self.runner.start()
        
    def waitForConnected(self):
        connected = False
        while not connected:
            if self.muse.client:
                if self.muse.client.is_connected:
                    print(f"Connected")
                    connected = True
                else:
                    print(f"Waiting for Muse")
            else:
                print("Muse client not yet initialized.")
            sleep(1)
        
    def disconnect(self):
        future = asyncio.run_coroutine_threadsafe(self.muse.stop(), self.runner.loop)
        try:
            future.result(timeout=5)
        except Exception:
            pass
        
        
class MuseBleak:
    def __init__(self, address, callback_eeg, time_func=time,
                 telemetry_callback=None, hsi_callback=None):
        self.address = address
        self.callback_eeg = callback_eeg
        self.callback_telemetry = telemetry_callback
        self.callback_hsi = hsi_callback
        self.time_func = time_func
        self.should_run = True
        self.client = None
        self.sample_index = 0
        self.last_tm = None
        self.timestamps = np.zeros(4)
        self.data = np.zeros((4, 12))
        self.reg_params = np.array([self.time_func(), 1. / SAMPLE_RATE])
        self._control_buffer = ''   # accumulates partial JSON from control stream
        self._status_task    = None  # asyncio task for periodic HSI polling

        self.pending_samples = {}  # {packet_index: {channel_index: (timestamp, data)}}
        self.expected_channels = {0, 1, 2, 3}

    async def connect_loop(self):
        """Loop to keep trying to connect and resume streaming."""
        while self.should_run:
            try:
                self._control_buffer = ''   # reset on each (re)connect
                await self.connect()
                print("Connected successfully.")
                while self.client.is_connected and self.should_run:
                    await asyncio.sleep(1)
                print("Disconnected. Reconnecting...")
                # Cancel the status polling task before reconnecting
                if self._status_task and not self._status_task.done():
                    self._status_task.cancel()
                self.last_tm = None

            except Exception as e:
                print(f"Connection error: {e}")
            await asyncio.sleep(3)  # Wait before trying to reconnect


    async def connect(self):
        self.client = BleakClient(self.address)
        await self.client.connect()

        # ── EEG channels ─────────────────────────────────────────────────────
        for i, name in enumerate(["TP9", "AF7", "AF8", "TP10"]):
            await self.client.start_notify(
                MUSE_GATT_ATTRS[name],
                lambda sender, data, i=i: asyncio.create_task(self.handle_eeg(i, data))
            )

        # ── Telemetry (battery) ───────────────────────────────────────────────
        if self.callback_telemetry is not None:
            await self.client.start_notify(
                MUSE_GATT_ATTRS["TELEMETRY"],
                lambda sender, data: asyncio.create_task(self.handle_telemetry(data))
            )

        # ── Control stream (HSI / horseshoe indicator) ────────────────────────
        if self.callback_hsi is not None:
            await self.client.start_notify(
                MUSE_GATT_ATTRS["STREAM_TOGGLE"],
                lambda sender, data: asyncio.create_task(self.handle_control(data))
            )

        await self.resume()

    async def resume(self):
        # 'd' command – start/resume streaming
        await self.client.write_gatt_char(
            MUSE_GATT_ATTRS["STREAM_TOGGLE"], bytearray([0x02, 0x64, 0x0a])
        )
        # 's' command – request a status dump (triggers the first HSI message)
        if self.callback_hsi is not None:
            await asyncio.sleep(0.1)
            await self.client.write_gatt_char(
                MUSE_GATT_ATTRS["STREAM_TOGGLE"], bytearray([0x02, 0x73, 0x0a])
            )
            print("[PROXY] Initial status request sent.")
            # Kick off periodic polling so HSI stays fresh
            self._status_task = asyncio.create_task(self._poll_status_loop())

    async def _poll_status_loop(self):
        """Re-send the 's' status request every 5 s to keep HSI updates flowing."""
        try:
            while self.should_run and self.client and self.client.is_connected:
                await asyncio.sleep(5)
                if not (self.should_run and self.client and self.client.is_connected):
                    break
                await self.client.write_gatt_char(
                    MUSE_GATT_ATTRS["STREAM_TOGGLE"], bytearray([0x02, 0x73, 0x0a])
                )
                print("[PROXY] Periodic status request sent.")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[PROXY] Poll-status error: {e}")

    async def stop(self):
        self.should_run = False
        if self.client:
            if self.client.is_connected:
                try:
                    await self.client.write_gatt_char(
                        MUSE_GATT_ATTRS["STREAM_TOGGLE"],
                        bytearray([0x02, 0x68, 0x0a])
                    )
                    await self.client.disconnect()
                    print("Requested client disconnect.")
                    # Wait briefly to allow disconnect to propagate
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"BLE disconnect failed: {e}")
            
            # Double-check if still connected (common on Linux)
            if self.client.is_connected:
                print("Client still connected after disconnect attempt.")
                force_disconnect(self.address)
            else:
                print("Client disconnected cleanly.")

    def unpack_eeg(self, packet):
        aa = bitstring.Bits(bytes=packet)
        pattern = "uint:16," + ",".join(["uint:12"] * 12)
        res = aa.unpack(pattern)
        packet_index, data = res[0], res[1:]
        data = 0.48828125 * (np.array(data) - 2048)
        return packet_index, data

    async def handle_telemetry(self, data: bytearray):
        """Parse battery level from telemetry packet.

        Muse telemetry packet layout (big-endian):
            bytes 0-1 : sequence ID
            bytes 2-3 : battery raw uint16  →  pct = raw / 512.0   (0–100)
            bytes 4-5 : fuel-gauge voltage  →  mV  = raw * 2.2
            bytes 8-9 : temperature (°C)
        """
        try:
            if len(data) < 4:
                return
            raw         = int.from_bytes(data[2:4], byteorder='big', signed=False)
            battery_pct = raw / 512.0          # muse-js formula – already gives 0-100
            battery_pct = max(0.0, min(100.0, battery_pct))
            print(f"[BATTERY] raw={raw}  pct={battery_pct:.1f}%")
            if self.callback_telemetry:
                self.callback_telemetry(battery_pct)
        except Exception as e:
            print(f"[TELEMETRY] Parse error: {e}")

    async def handle_control(self, data: bytearray):
        """Accumulate control-stream chars and extract HSI from JSON messages.

        Muse BLE control packet layout:
            byte 0      : 0x14 (20) – fixed MTU marker
            byte 1      : N – number of valid ASCII bytes in this chunk
            bytes 2..2+N: ASCII content (rest is null-padded to 20 bytes)

        Multiple chunks are concatenated until a complete JSON object is found::

            {"hs":[1,1,2,4], "batt":75, ...}

        HSI values: 1 = good contact, 2 = medium, 4 = bad.
        """
        import json
        try:
            if len(data) < 2:
                return
            # byte 1 = number of valid chars in this notification chunk
            n_chars = data[1]
            chunk = data[2: 2 + n_chars].decode('utf-8', errors='replace')
            self._control_buffer += chunk
            print(f"[CONTROL] chunk={repr(chunk)}  buf={repr(self._control_buffer)}")
            # Extract any complete JSON objects from the buffer
            while True:
                start = self._control_buffer.find('{')
                end   = self._control_buffer.find('}')
                if start == -1 or end == -1 or end < start:
                    break
                candidate = self._control_buffer[start : end + 1]
                self._control_buffer = self._control_buffer[end + 1:]
                try:
                    obj = json.loads(candidate)
                    print(f"[CONTROL] parsed JSON: {obj}")
                    if 'hs' in obj and self.callback_hsi:
                        self.callback_hsi(obj['hs'])   # list of 4 ints
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"[CONTROL] Parse error: {e}")

    async def handle_eeg(self, channel_index, data):
        timestamp = self.time_func()
        packet_index, channel_data = self.unpack_eeg(data)

        # Store the data in a buffer for this packet index
        if packet_index not in self.pending_samples:
            self.pending_samples[packet_index] = {}

        self.pending_samples[packet_index][channel_index] = (timestamp, channel_data)

        # If all 4 channels have arrived for this packet index
        if self.expected_channels.issubset(self.pending_samples[packet_index].keys()):
            # Sort by channel index (0 to 3)
            sorted_data = [self.pending_samples[packet_index][i][1] for i in range(4)]
            timestamps = [self.pending_samples[packet_index][i][0] for i in range(4)]
            timestamps = np.mean(timestamps)  # Approximate per-sample timestamp

            # Only process sequentially — skip out-of-order packets
            if self.last_tm is None or packet_index == (self.last_tm + 1) % 65536:
                self.last_tm = packet_index
                idxs = np.arange(12) + self.sample_index
                self.sample_index += 12
                exact_timestamps = self.reg_params[1] * idxs + self.reg_params[0]
                self.callback_eeg(exact_timestamps, np.transpose(np.array(sorted_data)))
            else:
                print(f"Dropped frame: got {packet_index}, expected {self.last_tm + 1}")

            # Clean up
            del self.pending_samples[packet_index]


async def main(muse):
    try:
        await muse.connect_loop()
    except asyncio.CancelledError:
        print("Async task cancelled.")
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        print("Shutting down MuseBleak...")
        await muse.stop()


def force_disconnect(address):
    try:
        print(f"Forcing disconnect of {address} using bluetoothctl...")
        subprocess.run(["bluetoothctl", "disconnect", address], check=True)
    except Exception as e:
        print(f"bluetoothctl disconnect failed: {e}")


class AsyncRunnerThread:
    def __init__(self, coro_func, *args):
        self.coro_func = coro_func
        self.args = args
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.task = None
        
    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.task = self.loop.create_task(self.coro_func(*self.args))
        try:
            self.loop.run_forever()
        finally:
            self._shutdown_loop()

    def _shutdown_loop(self):
        # Gracefully cancel the main task
        if self.task:
            self.loop.run_until_complete(self._cancel_task(self.task))
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()

    async def _cancel_task(self, task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print("Main async task cancelled.")

    def start(self):
        self.thread.start()

    def stop(self):
        try:
            if not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(timeout=5)
        except RuntimeError:
            pass  # loop already closed – nothing to do