"""
MoCap Streaming Module for an OptiTrack System running on Motive
Contains wrapper class MoCapStream, that wraps the NatNetSDK Python Client

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

from .NatNetSDK import NatNetClient
from datetime import timedelta
from collections import deque
import time
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

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
        
        # Set up listener
        self.client.new_frame_with_data_listener = self.new_frame_with_data_listener

        time.sleep(1)  # Allow some time to initialize

        # Check if the client is connected
        if not self.client.connected():
            raise RuntimeError(f"Failed to initialize MoCapStream: Could not connect to server {self.server_ip} at client {self.client_ip}.")

    def start_timing(self):
        self.initial_timing_offset = self.pose_buffer[-1]['timestamp']

    def resync_timing(self):
        self.initial_timing_offset += self.pose_buffer[-1]['timestamp']

    def new_frame_with_data_listener(self, data_dict):
        mocap_data = data_dict["mocap_data"]
        return_dict = {}

        rigid_body_list = mocap_data.rigid_body_data.rigid_body_list
        for rb in rigid_body_list:
            if rb.id_num == 2:
                return_dict["rigid_body_pose"] = {}
                return_dict["rigid_body_pose"]["position"] = list(rb.pos)
                return_dict["rigid_body_pose"]["rotation"] = list(rb.rot)
                return_dict["tracking_valid"] = rb.tracking_valid
                return_dict["mean_error"] = rb.error

        suffix_data = mocap_data.suffix_data
        timestamp = suffix_data.timestamp
        stamp_camera_mid_exposure = suffix_data.stamp_camera_mid_exposure
        stamp_data_received = suffix_data.stamp_data_received
        stamp_transmit = suffix_data.stamp_transmit

        return_timestamp = timedelta(seconds=(stamp_camera_mid_exposure / 10_000_000))
        
        # Apply timing offset if set
        if self.initial_timing_offset is not None:
            return_timestamp = return_timestamp - self.initial_timing_offset

        return_dict["timestamp"] = return_timestamp

        self.pose_buffer.append(return_dict)

    def getnext(self):
        return self.pose_buffer[-1].copy()
    
    def get_current_buffer(self):
        return self.pose_buffer.copy()
    
    def stop(self):
        self.client.shutdown()