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
from scipy.spatial.transform import Rotation as R, RotationSpline
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
        self.initial_timing_offset = None
        self.timestamp = None # Extra member needed because timestamp is retrieved via another listener
        self.pose_buffer = deque(maxlen=self.buffer_size)
        
        # Initialize the NatNet client
        self.client = NatNetClient()
        self.client.set_client_address(self.client_ip)
        self.client.set_server_address(self.server_ip)
        self.client.set_use_multicast(False)
        self.client.set_print_level(0)  # print_level = 0 off, print_level = 1 on, print_level = >1 on / print every nth mocap frame:
                                        # Addionally, comment out line 1663 in NatNetClient.py
        self.client.run("d")
        
        # Set up listeners for rigid_bodies and frames (needed for time sync) --> No threading needed because the NatNet client handles this internally
        self.client.rigid_body_listener = self.rigid_body_listener
        self.client.new_frame_listener = self.new_frame_listener

        time.sleep(1)  # Allow some time to initialize

        # Check if the client is connected
        if not self.client.connected():
            raise RuntimeError(f"Failed to initialize MoCapStream: Could not connect to server {self.server_ip} at client {self.client_ip}.")

    def start_timing(self):
        self.initial_timing_offset = self.pose_buffer[-1]['timestamp']

    def resync_timing(self):
        self.initial_timing_offset += self.pose_buffer[-1]['timestamp']

    def new_frame_listener(self, frame_data):
        timestamp = frame_data.get('timestamp')
        self.timestamp = timedelta(seconds=timestamp)

    def rigid_body_listener(self, rigid_body_id, position, rotation, marker_error, tracking_valid):
        if rigid_body_id == self.rigid_body_id:
            
            # Apply timing offset if set
            if self.initial_timing_offset is not None:
                timestamp = self.timestamp - self.initial_timing_offset
            else:
                timestamp = self.timestamp

            self.pose_buffer.append({
                'rigid_body_pose': {
                    'position': list(position),
                    'rotation': list(rotation),
                    },
                'timestamp': timestamp,
                'mean_error': marker_error,
                'tracking_valid': tracking_valid
            })

    def getnext(self):
        return self.pose_buffer[-1].copy()
    
    def get_current_buffer(self):
        return self.pose_buffer.copy()
    
    def stop(self):
        self.client.shutdown()