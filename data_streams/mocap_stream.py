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
                    'position': list(position),
                    'rotation': list(rotation),
                    },
                'mean_error': marker_error,
                'tracking_valid': tracking_valid
            })

    def get_current_data(self):
        return self.result_dict_buffer[-1].copy()
    
    def wait_for_n_poses(self, n):
        # Wait until the buffer has accumulated at least n new mocap poses
        future_poses_cnt = 0
        last_mocap_ts = self.result_dict_buffer[-1]['timestamp'].total_seconds()
        while True:
            current_mocap_ts = self.result_dict_buffer[-1]['timestamp'].total_seconds()
            if current_mocap_ts > last_mocap_ts:
                future_poses_cnt += 1
                last_mocap_ts = current_mocap_ts
            if future_poses_cnt >= n:
                break
            time.sleep(0.001)
    
    def get_interpolated_pose(self, query_cam_data, marker_error_threshold, show_plot=False):
        """
        Interpolates the poses in the buffer to find the pose at the timestamp of the camera data.
        
        Args:
            query_cam_data (dict): The camera data containing the timestamp to query.
            marker_error_threshold (float): The threshold for the marker error to consider a pose valid.
        
        Returns:
            numpy.ndarray: The interpolated position.
            numpy.ndarray: The interpolated rotation as a quaternion.
        """
        self.wait_for_n_poses(n=self.buffer_size // 2) # Ensure the buffer has enough poses to match
        current_buffer = self.result_dict_buffer.copy() # Get the current buffer of mocap data
        
        # Filter the buffer for valid mocap data based on tracking validity and marker error threshold
        interest_buffer = []
        for data in current_buffer:
            if data['tracking_valid'] and data['mean_error'] < marker_error_threshold:
                interest_buffer.append(data)

        if len(interest_buffer) < 5:
            print("Not enough valid mocap data in buffer to match with camera data.")
            return None, None
        
        # Extract times, positions, and rotations from the buffer
        times = [data['timestamp'].total_seconds() for data in interest_buffer]
        positions = [data['rigid_body_pose']['position'] for data in interest_buffer]
        rotations = [data['rigid_body_pose']['rotation'] for data in interest_buffer]

        # Create Translation-Splines (one for each xyz-dim) from valid buffer
        positions_plot = positions.copy()
        positions = np.array(positions).T
        pos_splines = [CubicSpline(times, positions[dim]) for dim in range(3)]

        # Create Rotation-Spline from valid buffer
        rotations_plot = rotations.copy()
        rotations = R.from_quat(rotations)
        rot_spline = RotationSpline(times, rotations)

        # Query
        query_time = query_cam_data['timestamp'].total_seconds()
        interpolated_position = np.array([s(query_time) for s in pos_splines])
        interpolated_rotation = rot_spline(query_time).as_quat()

        # Plotting for debugging (optional)
        if show_plot:
            times_plot = np.linspace(times[0], times[-1], 100)
            rot_spline_plot = rot_spline(times_plot).as_quat()

            fig, axs = plt.subplots(2, 1, figsize=(10, 8))

            # Plot quaternion components
            axs[0].plot(times_plot, rot_spline_plot)
            axs[0].plot(times, rotations_plot, 'x', color='black')
            query_line0 = axs[0].axvline(query_time, color='red', linestyle='--')
            axs[0].set_title("Quaternions over time")
            axs[0].set_xlabel("Time [s]")
            axs[0].set_ylabel("Quaternion components")
            # Custom legend handles
            handles0 = [
                mlines.Line2D([], [], color=axs[0].lines[i].get_color(), label=lbl)
                for i, lbl in enumerate(['w', 'x', 'y', 'z'])
            ]
            handles0.append(
                mlines.Line2D([], [], color='red', linestyle='--', label='query_time')
            )
            axs[0].legend(handles=handles0)

            # Plot position splines
            pos_spline_plot = np.array([[s(t) for s in pos_splines] for t in times_plot])
            axs[1].plot(times_plot, pos_spline_plot)
            axs[1].plot(times, positions_plot, 'x', color='black')
            query_line1 = axs[1].axvline(query_time, color='red', linestyle='--')
            axs[1].set_title("Position XYZ over time")
            axs[1].set_xlabel("Time [s]")
            axs[1].set_ylabel("Position [m]")
            handles1 = [
                mlines.Line2D([], [], color=axs[1].lines[i].get_color(), label=lbl)
                for i, lbl in enumerate(['x', 'y', 'z'])
            ]
            handles1.append(
                mlines.Line2D([], [], color='red', linestyle='--', label='query_time')
            )
            axs[1].legend(handles=handles1)

            plt.tight_layout()
            plt.show()

        return interpolated_position, interpolated_rotation