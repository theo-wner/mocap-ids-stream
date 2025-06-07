"""
Camera streaming module for an IDS camera.
Contains wrapper class CameraStream, that wraps the IDS Peak API

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import threading
import cv2
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension

class CameraStream:
    """
    A class to stream images from an IDS camera in the background.

    Attributes:
        frame (numpy.ndarray): The latest captured image frame.
        running (bool): Streaming status flag.
        frame_rate (int): Frame rate for the camera stream.
        exposure_time (double): Exposure time in microseconds.
        resize (tuple): Resize dimensions for the output image.
        thread (threading.Thread): Thread for camera streaming.
    """

    def __init__(self, frame_rate=30, exposure_time=10000, resize=(500, 500)):
        """
        Initializes the CameraStream class and starts the camera stream in a separate thread.

        Args:
            frame_rate (int): Frame rate for the camera stream.
            exposure_time (double): Exposure time in microseconds.
            resize (tuple): Resize dimensions for the output image.
        """
        self.frame = None
        self.timestamp = None
        self.running = True
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.resize = resize

        # Initialize camera streaming in a separate thread
        self.thread = threading.Thread(target=self.camera_loop)
        self.thread.daemon = True
        self.thread.start()

    def camera_loop(self):
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
                    buffer = data_stream.WaitForFinishedBuffer(1000)
                    if buffer.HasChunks():
                        remote_nodemap.UpdateChunkNodes(buffer)
                        timestamp = remote_nodemap.FindNode("ChunkTimestamp").Value()

                    img = ids_peak_ipl_extension.BufferToImage(buffer)

                    frame_bayer = img.get_numpy_2D()
                    frame = cv2.cvtColor(frame_bayer, cv2.COLOR_BAYER_BG2BGR)

                    if self.resize:
                        frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)

                    self.frame = frame
                    self.timestamp = timestamp
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

    def get_current_frame(self):
        """
        Returns the latest image frame captured by the camera.

        Returns:
            numpy.ndarray or None: The latest image frame, or None if not yet available.
        """
        return (self.frame, self.timestamp)

    def stop(self):
        """
        Stops the camera stream and releases resources.
        """
        self.running = False
        self.thread.join()

