"""
Camera streaming module for an IDS camera.
Contains wrapper class IDSStream, that wraps the IDS Peak API

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import threading
import cv2
import torch
import time
from datetime import timedelta
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension

class IDSStream:
    """
    A class to stream images from an IDS camera in the background.
    """

    def __init__(self, frame_rate, exposure_time, white_balance, gain, gamma, resize):
        # Member variables to control the streaming
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.white_balance = white_balance
        self.gain = gain
        self.gamma = gamma
        self.resize = resize

        # Member variables to store the latest data
        self.initial_timing_offset = None
        self.frame = None
        self.info = {"timestamp": None, "is_test": False}

        # Initialize camera streaming in a separate thread
        self.running = True
        self.thread = threading.Thread(target=self.update_loop, daemon=True)
        self.thread.start()

        time.sleep(1)  # Allow some time for the camera to initialize

        # Check if the camera is initialized
        if self.frame is None:
            raise RuntimeError("Failed to initialize IDSStream: Camera not found or not initialized.")

    def __len__(self):
        # Arbitrary large number as we don't know the length of a stream
        return 100_000_000  

    def start_timing(self):
        self.initial_timing_offset = self.info["timestamp"]

    def resync_timing(self):
        self.initial_timing_offset += self.info["timestamp"]

    def update_loop(self):
        """
        Internal method to handle camera initialization and image streaming.
        Inspired by the example from the IDS peak library at https://pypi.org/project/ids-peak/
        """
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()

        try:
            device_manager.Update()
            if device_manager.Devices().empty():
                print("No IDS camera found.")
                self.running = False
                return

            device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            remote_nodemap = device.RemoteDevice().NodeMaps()[0]

            # Set acquisition parameters
            # Frame rate
            if self.frame_rate == 'max':
                max_frame_rate = remote_nodemap.FindNode("AcquisitionFrameRate").Maximum()
                remote_nodemap.FindNode("AcquisitionFrameRate").SetValue(max_frame_rate)
                print(f"Using maximum frame rate: {max_frame_rate} FPS")
            elif isinstance(self.frame_rate, int):
                remote_nodemap.FindNode("AcquisitionFrameRate").SetValue(self.frame_rate)
            else:
                self.running = False
                raise ValueError("Possible values for frame_rate are 'max' or (int)")

            # Exposure time
            if self.exposure_time == 'auto':
                remote_nodemap.FindNode("ExposureAuto").SetCurrentEntry("Continuous")
            elif isinstance(self.exposure_time, int):
                remote_nodemap.FindNode("ExposureTime").SetValue(self.exposure_time)
            else:
                self.running = False
                raise ValueError("Possible values for exposure_time are 'auto' or (int)")

            # White balance
            if self.white_balance == 'auto':
                remote_nodemap.FindNode("BalanceWhiteAuto").SetCurrentEntry("Continuous")
            elif self.white_balance == 'off':
                 remote_nodemap.FindNode("BalanceWhiteAuto").SetCurrentEntry("Off")
            else:    
                self.running = False
                raise ValueError("Possible values for white_balance are 'auto' or 'off'")

            # Gain --> similar to ISO
            if self.gain == 'auto':
                remote_nodemap.FindNode("GainAuto").SetCurrentEntry("Continuous")
            elif self.gain == 'manually':
                remote_nodemap.FindNode("GainAuto").SetCurrentEntry("Off")
                manual_gains = {"DigitalRed": 1.0, "DigitalGreen": 1.0, "DigitalBlue": 1.0}
                for channel, gain_value in manual_gains.items():
                    remote_nodemap.FindNode("GainSelector").SetCurrentEntry(channel)
                    remote_nodemap.FindNode("Gain").SetValue(gain_value)
            else:    
                self.running = False
                raise ValueError("Possible values for gain are 'auto' or 'manually'")       

            # Gamma-correction
            remote_nodemap.FindNode("Gamma").SetValue(self.gamma)

            # Enable Metadata (Chunks) for timestamp retrieval
            remote_nodemap.FindNode("ChunkModeActive").SetValue(True)
            remote_nodemap.FindNode("ChunkSelector").SetCurrentEntry("Timestamp")
            remote_nodemap.FindNode("ChunkEnable").SetValue(True)

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

            while self.running:
                try:
                    # Process image data
                    buffer = data_stream.WaitForFinishedBuffer(1000)
                    remote_nodemap.UpdateChunkNodes(buffer)
                    frame = ids_peak_ipl_extension.BufferToImage(buffer)
                    frame = frame.get_numpy_2D()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR) # Convert Bayer pattern to BGR format
                    if self.resize:
                        frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)
                    self.frame = frame

                    # Process timestamp
                    timestamp = remote_nodemap.FindNode("ChunkTimestamp").Value()
                    timestamp = timedelta(seconds=timestamp / 1e9)  # Convert nanoseconds to seconds
                    if self.initial_timing_offset is not None:
                        timestamp = timestamp - self.initial_timing_offset
                    self.info['timestamp'] = timestamp
                    self.info['is_test'] = False 

                    # Queue the buffer for reuse
                    data_stream.QueueBuffer(buffer)
                except Exception as e:
                    print(f"Streaming exception: {e}")
                    break

            # Stop stream
            remote_nodemap.FindNode("AcquisitionStop").Execute()
            remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()
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
            print("Camera stream stopped.")
    
    def getnext(self, return_tensor=True):
        """
        Returns the next frame and its metadata.
        """    
        frame = self.frame.copy()
        info = self.info.copy()
        if return_tensor:
            frame = torch.from_numpy(frame).permute(2, 0, 1).cuda().float() / 255.0
        return frame, info
    
    def get_image_size(self):
        frame = self.getnext()[0]
        return frame.shape[-2], frame.shape[-1]

    def stop(self):
        self.running = False
        self.thread.join()
