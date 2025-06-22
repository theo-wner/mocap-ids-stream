"""
Module for matching camera frames with MoCap poses based on timestamps and quality filters.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import time
from collections import deque
import threading

class StreamMatcher:
    """
    A class to match camera frames with MoCap poses based on timestamps and quality filters.
    """

    def __init__(self, camera_stream, mocap_stream, maxlen=300):
        """
        Initializes the StreamMatcher class.

        Args:
            camera_stream (CameraStream): An instance of the CameraStream class.
            mocap_stream (MoCapStream): An instance of the MoCapStream class.
            maxlen (int): Maximum length of the buffer for mocap data.
        """
        self.camera_stream = camera_stream
        self.mocap_stream = mocap_stream
        self.cam_buffer = deque(maxlen=maxlen)
        self.mocap_buffer = deque(maxlen=maxlen)
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _update_loop(self):
        last_cam_ts = None
        last_mocap_ts = None
        while self.running:
            cam_data = self.camera_stream.get_current_data()
            mocap_data = self.mocap_stream.get_current_data()
            if cam_data and cam_data['timestamp'] is not None and cam_data['timestamp'] != last_cam_ts:
                self.cam_buffer.append(cam_data)
                last_cam_ts = cam_data['timestamp']
            if mocap_data and mocap_data['timestamp'] is not None and mocap_data['timestamp'] != last_mocap_ts:
                self.mocap_buffer.append(mocap_data)
                last_mocap_ts = mocap_data['timestamp']
            time.sleep(0.001)

    def get_best_match(self, marker_error_threshold=None, require_tracking_valid=None):
        """
        Waits until there are at least buffer_n_poses mocap poses after the latest camera frame,
        then returns the best match based on timestamp proximity and optional quality filters.
        """
        buffer_n_poses = self.mocap_buffer.maxlen // 2  # Use half of the buffer size for matching

        cam = self.cam_buffer[-1]
        cam_ts = cam['timestamp'].total_seconds()

        # Wait for enough future mocap poses
        while True:
            mocap_timestamps = [
                m['timestamp'].total_seconds()
                for m in self.mocap_buffer
            ]
            after_count = sum(ts > cam_ts for ts in mocap_timestamps)
            if after_count >= buffer_n_poses:
                break
            time.sleep(0.001)

        best_mocap = None
        best_dt = float('inf')
        cnt=0
        for mocap in self.mocap_buffer:
            mocap_ts = mocap['timestamp'].total_seconds() if hasattr(mocap['timestamp'], "total_seconds") else mocap['timestamp']
            if marker_error_threshold is not None and (mocap['mean_error'] is None or mocap['mean_error'] > marker_error_threshold):
                continue
            if require_tracking_valid is not None and mocap['tracking_valid'] != require_tracking_valid:
                continue
            dt = abs(cam_ts - mocap_ts)
            if dt < best_dt:
                best_dt = dt
                best_mocap = mocap
            cnt += 1
        print(f'Position in buffer: {cnt}')
        return cam, best_mocap, best_dt if best_mocap is not None else (cam, None, None)