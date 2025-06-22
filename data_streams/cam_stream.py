"""
Camera streaming module for an IDS camera.
Contains wrapper class CamStream, that wraps the IDS Peak API

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import threading
import cv2
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
from datetime import timedelta

class CamStream:
    """
    A class to stream images from an IDS camera in the background.
    """

    def __init__(self, frame_rate=30, exposure_time=10000, resize=(500, 500)):
        # Member variables to control the streaming
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.resize = resize
        self.running = False

        # Member variables to store the latest data
        self.timing_offset = None
        self.result_dict = {"timestamp": None, "frame": None}

    def start(self):
        # Initialize camera streaming in a separate thread
        self.running = True
        self.thread = threading.Thread(target=self.update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def start_timing(self):
        self.timing_offset = self.result_dict["timestamp"]

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
            remote_nodemap.FindNode("AcquisitionFrameRate").SetValue(self.frame_rate)
            remote_nodemap.FindNode("ExposureTime").SetValue(self.exposure_time)

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
                        self.result_dict["frame"] = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)
                    else:
                        self.result_dict["frame"] = frame

                    # Process timestamp
                    timestamp = remote_nodemap.FindNode("ChunkTimestamp").Value()
                    timestamp = timedelta(seconds=timestamp / 1e9)  # Convert nanoseconds to seconds
                    if self.timing_offset is not None:
                        timestamp = timestamp - self.timing_offset
                    self.result_dict["timestamp"] = timestamp

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
            ids_peak.Library.Close()
            print("Camera stream stopped.")

    def get_current_data(self):
        return self.result_dict

