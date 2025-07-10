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
        self.pose_buffer = deque(maxlen=self.buffer_size)
        
        # Initialize the NatNet client
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

        time.sleep(1) # Allow some time for the client to connect and start receiving data

    def start_timing(self):
        self.timing_offset = self.pose_buffer[-1]['timestamp']

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
    
    def wait_for_n_poses(self, n):
        """
        Waits until the buffer has at least n future poses.
        """
        future_poses_cnt = 0
        last_mocap_ts = self.pose_buffer[-1]['timestamp'].total_seconds()
        while True:
            current_mocap_ts = self.pose_buffer[-1]['timestamp'].total_seconds()
            if current_mocap_ts > last_mocap_ts:
                future_poses_cnt += 1
                last_mocap_ts = current_mocap_ts
            if future_poses_cnt >= n:
                break
            time.sleep(0.001)
    
    def get_interpolated_pose(self, query_time, marker_error_threshold, show_plot=False):
        """
        Interpolates the poses in the buffer to find the pose at the timestamp of the camera data.
        
        Args:
            query_time (timedelta): The timestamp at which to query the pose.
            marker_error_threshold (float): The threshold for the marker error to consider a pose valid.
        
        Returns:
            numpy.ndarray: The interpolated position.
            numpy.ndarray: The interpolated rotation as a quaternion.
        """
        self.wait_for_n_poses(n=self.buffer_size // 2) # Ensure the buffer has enough poses to match
        current_buffer = self.pose_buffer.copy() # Get the current buffer of mocap data
        
        # Filter the buffer for valid mocap data based on tracking validity and marker error threshold
        interest_buffer = []
        for data in current_buffer:
            if data['tracking_valid'] and data['mean_error'] < marker_error_threshold:
                interest_buffer.append(data)

        if len(interest_buffer) < 5:
            return None, None, None, None
        
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

        # Query pose at the camera timestamp
        query_time = query_time.total_seconds()
        interpolated_position = np.array([s(query_time) for s in pos_splines])
        interpolated_rotation = rot_spline(query_time).as_quat()

        # Query velocities at the camera timestamp
        interpolated_linear_velocity_vec = np.array([s(query_time, 1) for s in pos_splines])
        interpolated_lateral_velocity = np.linalg.norm(interpolated_linear_velocity_vec)

        interpolated_angular_velocity_vec = rot_spline(query_time, 1)
        interpolated_angular_velocity = np.linalg.norm(interpolated_angular_velocity_vec)

        # Plotting for debugging (optional)
        if show_plot:
            times_plot = np.linspace(times[0], times[-1], 100)
            rot_spline_plot = rot_spline(times_plot).as_quat()

            # Compute velocities for the whole plot range
            linear_velocities = np.array([[s(t, 1) for s in pos_splines] for t in times_plot])
            lateral_velocities = np.linalg.norm(linear_velocities, axis=1)
            angular_velocities = np.linalg.norm(rot_spline(times_plot, 1), axis=1)

            fig, axs = plt.subplots(3, 1, figsize=(10, 12))

            # Plot quaternion components
            axs[0].plot(times_plot, rot_spline_plot)
            axs[0].plot(times, rotations_plot, 'x', color='black')
            query_line0 = axs[0].axvline(query_time, color='red', linestyle='--')
            axs[0].set_title("Quaternions over time")
            axs[0].set_xlabel("Time [s]")
            axs[0].set_ylabel("Quaternion components")
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

            # Plot velocities
            axs[2].plot(times_plot, lateral_velocities, label='Lateral velocity (m/s)', color='blue')
            axs[2].plot(times_plot, angular_velocities, label='Angular velocity (rad/s)', color='orange')
            axs[2].axvline(query_time, color='red', linestyle='--', label='query_time')
            axs[2].scatter([query_time], [interpolated_lateral_velocity], color='blue', marker='x')
            axs[2].scatter([query_time], [interpolated_angular_velocity], color='orange', marker='x')
            axs[2].set_title("Velocities over time")
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("Velocity")
            axs[2].legend()

            plt.tight_layout()
            plt.show()

        return interpolated_position, interpolated_rotation, interpolated_lateral_velocity, interpolated_angular_velocity
    
    def stop(self):
        self.client.shutdown()