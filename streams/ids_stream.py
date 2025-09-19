"""
Camera streaming module for an IDS camera.
Contains wrapper class IDSStream, that wraps the IDS Peak API

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import cv2
import numpy as np
import time
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import scipy.interpolate as interp


class IDSStream:
    """
    A class to stream images from an IDS camera in the background.
    """

    def __init__(self, frame_rate, exposure_time, white_balance, gain, gamma):
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.white_balance = white_balance
        self.gain = gain
        self.gamma = gamma
        self.frame_shape = (2840, 2840)

        # Shared memory for frame
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.frame_shape) * np.uint8().nbytes)
        self.frame = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.shm.buf)

        # Control flag
        self.running = mp.Event()
        self.running.set()

        # Start camera process
        self.process = mp.Process(target=self.update_loop, daemon=True)
        self.process.start()

        # Wait for first frame
        start_time = time.time()
        first_frame = False
        while time.time() - start_time < 5:
            if np.any(self.frame):
                first_frame = True
                break
            time.sleep(0.05)
        if not first_frame:
            self.stop()
            raise RuntimeError("Camera not initialized in time")
        
    def get_image_size(self):
        return self.frame_shape

    def update_loop(self):
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()

        try:
            device_manager.Update()
            if device_manager.Devices().empty():
                print("No IDS camera found")
                self.running.clear()
                return

            device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            remote_nodemap = device.RemoteDevice().NodeMaps()[0]

            # Frame rate
            if self.frame_rate == 'max':
                remote_nodemap.FindNode("AcquisitionFrameRate").SetValue(
                    remote_nodemap.FindNode("AcquisitionFrameRate").Maximum())
            elif isinstance(self.frame_rate, int):
                remote_nodemap.FindNode("AcquisitionFrameRate").SetValue(self.frame_rate)
            else:
                raise ValueError("frame_rate must be 'max' or int")

            # Exposure
            if self.exposure_time == 'auto':
                remote_nodemap.FindNode("ExposureAuto").SetCurrentEntry("Continuous")
            elif isinstance(self.exposure_time, int):
                remote_nodemap.FindNode("ExposureTime").SetValue(self.exposure_time)
            else:
                raise ValueError("exposure_time must be 'auto' or int")

            # White balance
            if self.white_balance == 'auto':
                remote_nodemap.FindNode("BalanceWhiteAuto").SetCurrentEntry("Continuous")
            elif self.white_balance == 'off':
                remote_nodemap.FindNode("BalanceWhiteAuto").SetCurrentEntry("Off")
            else:
                raise ValueError("white_balance must be 'auto' or 'off'")

            # Gain
            if self.gain == 'auto':
                remote_nodemap.FindNode("GainAuto").SetCurrentEntry("Continuous")
            elif self.gain in ['default', 'manually']:
                remote_nodemap.FindNode("GainAuto").SetCurrentEntry("Off")
                if self.gain == 'manually':
                    manual_gains = {"DigitalRed":1.0,"DigitalGreen":1.0,"DigitalBlue":1.0}
                    for channel, val in manual_gains.items():
                        remote_nodemap.FindNode("GainSelector").SetCurrentEntry(channel)
                        remote_nodemap.FindNode("Gain").SetValue(val)
            else:
                raise ValueError("gain must be 'auto','default' or 'manually'")

            # Gamma
            remote_nodemap.FindNode("Gamma").SetValue(self.gamma)

            # Prepare data stream
            data_stream = device.DataStreams()[0].OpenDataStream()
            payload_size = remote_nodemap.FindNode("PayloadSize").Value()
            for _ in range(data_stream.NumBuffersAnnouncedMinRequired()):
                buffer = data_stream.AllocAndAnnounceBuffer(payload_size)
                data_stream.QueueBuffer(buffer)

            remote_nodemap.FindNode("TLParamsLocked").SetValue(1)
            data_stream.StartAcquisition()
            remote_nodemap.FindNode("AcquisitionStart").Execute()
            remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()
            print("Camera stream started.")

            while self.running.is_set():
                try:
                    buffer = data_stream.WaitForFinishedBuffer(1000)
                    img = ids_peak_ipl_extension.BufferToImage(buffer).get_numpy_2D()
                    
                    # Copy to shared memory
                    np.copyto(self.frame, img)

                    data_stream.QueueBuffer(buffer)
                except Exception as e:
                    print(f"Streaming exception: {e}")
                    break

            # Stop acquisition
            remote_nodemap.FindNode("AcquisitionStop").Execute()
            data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for buffer in data_stream.AnnouncedBuffers():
                data_stream.RevokeBuffer(buffer)
            remote_nodemap.FindNode("TLParamsLocked").SetValue(0)

        except Exception as e:
            print(f"Camera setup failed: {e}")
        finally:
            remote_nodemap.FindNode("DeviceReset").Execute()
            ids_peak.Library.Close()
            print("Camera stream stopped")

    def getnext(self):
        return cv2.cvtColor(self.frame, cv2.COLOR_BAYER_BG2BGR), time.time()
    
    def sync_event(self):
        times = []
        rows = []

        while True:
            frame, timestamp = self.getnext()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Threshold and erosion
            _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
            kernel = np.ones((25, 25), np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=1)

            # Draw circles only on the RGB frame
            for c in cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                if 5000 <= cv2.contourArea(c) <= 500000:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 10) 
                    times.append(timestamp)
                    rows.append(y)

            # Show
            eroded_bgr = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
            combined = cv2.hconcat([eroded_bgr, frame])
            combined_resized = cv2.resize(combined, (1200, 600))
            cv2.imshow("Eroded | RGB", combined_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        times = np.array(times)
        rows = np.array(rows)

        _, unique_indices = np.unique(rows, return_index=True)
        unique_indices.sort()
        unique_rows = rows[unique_indices]
        unique_times = times[unique_indices]

        interest_idx = np.argmin(unique_rows)
        start_idx = interest_idx - 10
        end_idx = interest_idx + 11

        selected_rows = unique_rows[start_idx:end_idx]
        selected_times = unique_times[start_idx:end_idx]

        spline = interp.UnivariateSpline(selected_times, selected_rows, k=3, s=0)

        time_fine = np.linspace(selected_times[0], selected_times[-1], 200)
        rows_fine = spline(time_fine)

        interest_idx = np.argmin(rows_fine)
        interest_row = rows_fine[interest_idx]
        interest_time = time_fine[interest_idx]

        return {"interest_time" : interest_time,
                "interest_row" : interest_row,
                "original_times" : selected_times,
                "original_rows" : selected_rows,
                "interp_times" : time_fine,
                "interp_rows": rows_fine}

    def stop(self):
        self.running.clear()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.shm.close()
        self.shm.unlink()

