"""
MoCap Streaming Module for an OptiTrack System running on Motive
Contains wrapper class MoCapStream, that wraps the NatNetSDK Python Client

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import numpy as np
from .NatNetSDK import NatNetClient
from datetime import timedelta
from collections import deque
import time

class MoCapStream:
    """
    A class to stream motion capture data from a NatNet server.
    """

    def __init__(self, client_ip, server_ip, rigid_body_id, buffer_size):
        # Member variables to control the streaming
        self.client_ip = client_ip
        self.server_ip = server_ip
        self.rigid_body_id = rigid_body_id
        self.buffer_size = buffer_size

        # Member variables to buffer the data
        self.timing_offset = None
        self.timestamp = None # Extra member needed because timestamp is retrieved via another listener
        self.result_dict_buffer = deque(maxlen=self.buffer_size)
        
    def start(self):
        self.client = NatNetClient()
        self.client.set_client_address(self.client_ip)
        self.client.set_server_address(self.server_ip)
        self.client.set_use_multicast(False)
        self.client.set_print_level(0)  # print_level = 0 off, print_level = 1 on, print_level = >1 on / print every nth mocap frame:
                                        # Addionally, comment out line 1663 in NatNetClient.py
        if not self.client.run("d"):
            raise RuntimeError("Failed to start NatNet client.")
        
        # Set up listeners for rigid_bodies and frames (needed for time sync) --> No threading needed because the NatNet client handles this internally
        self.client.rigid_body_listener = self.rigid_body_listener
        self.client.new_frame_listener = self.new_frame_listener

    def stop(self):
        self.client.shutdown()

    def start_timing(self):
        self.timing_offset = self.result_dict_buffer[-1]['timestamp']

    def new_frame_listener(self, frame_data):
        timestamp = frame_data.get('timestamp')
        self.timestamp = timedelta(seconds=timestamp)

    def rigid_body_listener(self, rigid_body_id, position, rotation, marker_error, tracking_valid):
        if rigid_body_id == self.rigid_body_id:

            # Apply timing offset if set
            if self.timing_offset is not None:
                timestamp = self.timestamp - self.timing_offset
            else:
                timestamp = self.timestamp

            self.result_dict_buffer.append({
                'timestamp': timestamp,
                'rigid_body_pose': {
                    'position': np.array(position),
                    'rotation': np.array(rotation),
                    },
                'mean_error': marker_error,
                'tracking_valid': tracking_valid
            })

    def get_current_data(self):
        return self.result_dict_buffer[-1]

    def get_best_match(self, query_cam_data):
        half_buffer_size = self.buffer_size // 2
        cam_ts = query_cam_data['timestamp'].total_seconds()

        # Wait for future mocap poses
        future_poses_cnt = 0
        last_mocap_ts = self.result_dict_buffer[-1]['timestamp'].total_seconds()
        while True:
            current_mocap_ts = self.result_dict_buffer[-1]['timestamp'].total_seconds()
            if current_mocap_ts > last_mocap_ts:
                future_poses_cnt += 1
                last_mocap_ts = current_mocap_ts
            if future_poses_cnt >= half_buffer_size:
                break
            time.sleep(0.001)

        # Find the best matching mocap data based on timestamp
        interest_buffer = self.result_dict_buffer.copy()
        best_mocap_data = None
        best_dt = float('inf')
        for mocap_data in interest_buffer:
            mocap_ts = mocap_data['timestamp'].total_seconds()
            dt = abs(cam_ts - mocap_ts)
            if dt < best_dt:
                best_dt = dt
                best_mocap_data = mocap_data

        return best_mocap_data, best_dt
