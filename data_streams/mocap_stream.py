"""
MoCap Streaming Module for an OptiTrack System running on Motive
Contains wrapper class MoCapStream, that wraps the NatNetSDK Python Client

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import numpy as np
from .NatNetSDK import NatNetClient
from datetime import timedelta

class MoCapStream:
    """
    A class to stream motion capture data from a NatNet server.
    """

    def __init__(self, client_ip, server_ip, rigid_body_id):
        # Member variables to control the streaming
        self.client_ip = client_ip
        self.server_ip = server_ip
        self.rigid_body_id = rigid_body_id

        # Member variables to store the latest data
        self.timestamp = None
        self.rigid_body_pose = None
        self.mean_error = None
        self.tracking_valid = None
        self.timing_offset = None
        
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
        self.timing_offset = self.timestamp

    def new_frame_listener(self, frame_data):
        timestamp = frame_data.get('timestamp')
        self.timestamp = timedelta(seconds=timestamp)

    def rigid_body_listener(self, rigid_body_id, position, rotation, marker_error, tracking_valid):
        if rigid_body_id == self.rigid_body_id:
            self.mean_error = marker_error
            self.tracking_valid = tracking_valid

            self.rigid_body_pose = {
                'position': np.array(position),
                'rotation': np.array(rotation),
            }

    def get_current_data(self):
        if self.timing_offset is not None:
            timestamp = self.timestamp - self.timing_offset
        else:
            timestamp = self.timestamp
            
        return {'timestamp': timestamp,
                'rigid_body_pose': self.rigid_body_pose,
                'mean_error': self.mean_error,
                'tracking_valid': self.tracking_valid}



