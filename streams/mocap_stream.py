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
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

class MoCapStream:
    """
    A class to stream motion capture data from a NatNet server.
    """

    def __init__(self, client_ip, server_ip, buffer_size):
        # Member variables to control the streaming
        self.client_ip = client_ip
        self.server_ip = server_ip
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


        return_dict = {
            "rigid_bodies": {},
            "labeled_markers": {},
            "timestamp": None
        }

        # Rigid bodies
        rigid_body_list = mocap_data.rigid_body_data.rigid_body_list
        for rb in rigid_body_list:
            return_dict["rigid_bodies"][rb.id_num] = {
                "pos": list(rb.pos),
                "rot": list(rb.rot),
                "tracking_valid": rb.tracking_valid,
                "mean_error": rb.error,
            }

        # Labeled markers
        labeled_marker_list = mocap_data.labeled_marker_data.labeled_marker_list
        for lb in labeled_marker_list:
            return_dict["labeled_markers"][lb.id_num] = {
                "pos": list(lb.pos),
                "belongs_to_rb" : (lb.param) == 10
            }
        # Timestamp
        return_dict["timestamp"] = time.time()
        self.pose_buffer.append(return_dict)

    def getnext(self):
        return self.pose_buffer[-1].copy()
    
    def get_current_buffer(self):
        return self.pose_buffer.copy()

    def sync_event(self, stop_event=None):
        times = []
        ys = []

        while True:
            if stop_event is not None and stop_event.is_set():

                break
            try:        
                mocap_data = self.getnext()

                marker_list = list(mocap_data["labeled_markers"].values())
                single_markers = [m for m in marker_list if not m["belongs_to_rb"]]

                if len(single_markers) != 1:
                    continue

                y = single_markers[0]["pos"][1]
                ys.append(y)

                times.append(mocap_data["timestamp"])
                time.sleep(0.0001)

            except KeyboardInterrupt:
                break

        times = np.array(times)
        ys = np.array(ys)

        _, unique_indices = np.unique(ys, return_index=True)
        unique_indices.sort()
        unique_ys = ys[unique_indices]
        unique_times = times[unique_indices]

        interest_idx = np.argmax(unique_ys)
        start_idx = interest_idx - 20
        end_idx = interest_idx + 21

        selected_ys = unique_ys[start_idx:end_idx]
        selected_times = unique_times[start_idx:end_idx]

        spline = interp.UnivariateSpline(selected_times, selected_ys, k=3, s=0)

        time_fine = np.linspace(selected_times[0], selected_times[-1], 200)
        ys_fine = spline(time_fine)

        interest_idx = np.argmax(ys_fine)
        interest_y = ys_fine[interest_idx]
        interest_time = time_fine[interest_idx]

        return {"interest_time" : interest_time,
                "interest_y" : interest_y,
                "original_times" : selected_times,
                "original_ys" : selected_ys,
                "interp_times" : time_fine,
                "interp_ys": ys_fine}
    
    def stop(self):
        self.client.shutdown()