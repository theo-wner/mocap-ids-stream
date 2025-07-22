"""
Module for matching data and handling time sync of an IDS Camera Stream and a OptiTrack MoCap Stream

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""
import numpy as np
import threading
import time
import torch
from scipy.spatial.transform import Rotation as R, RotationSpline
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class StreamMatcher():
    """
    A class to handle Streams from both an IDS Camera and a OptiTrack MoCap System
    """
    def __init__(self, ids_stream, mocap_stream, resync_interval):
        self.ids_stream = ids_stream
        self.mocap_stream = mocap_stream
        self.resync_interval = resync_interval
        self.resync_thread = threading.Thread(target=self.resync_loop, daemon=True)
        self.running = True

    def start_timing(self):
        self.ids_stream.start_timing()
        self.mocap_stream.start_timing()
        self.resync_thread.start()
        time.sleep(1)

    def resync_loop(self):
        while self.running:
            time.sleep(self.resync_interval)
            print("Resyncronizing timestamps")
            self.ids_stream.start_timing
            self.mocap_stream.start_timing

    def wait_for_n_poses(self, n):
        # Waits until the pose buffer has at least n future poses.
        future_poses_cnt = 0
        last_mocap_ts = self.mocap_stream.getnext()['timestamp'].total_seconds()
        while True:
            current_mocap_ts = self.mocap_stream.getnext()['timestamp'].total_seconds()
            if current_mocap_ts > last_mocap_ts:
                future_poses_cnt += 1
                last_mocap_ts = current_mocap_ts
            if future_poses_cnt >= n:
                break
            time.sleep(0.0001)

    def getnext(self, return_tensor=True, for_image=None, show_plot=False):
        """
        Returns the current frame together with its corresponding pose information

        Args:
            - return_tensor (boolean): Determines wheather to return the image frame as pytorch tensor or not
            - for_image (tuple): Optional, if passed, takes image and info as tuple from the argument, if None, retrieves a new image from ids_stream 
            - show_plot (boolean): Determines wheather to show a plot for pose visulization or not

        Returns:
            - frame (numpy.ndarray): The image
            - info (dict): Dictionary containing the pose info
                - keys:
                    - is_valid (boolean): Shows if the interpolated pose is valid
                    - pose (dict): The interpolated pose by the timestamp of image retrieval
                    - pose_velocitiy (dict): The interpolated pose velocities by the timestamp of image retrieval
        """
        if for_image is None:
            frame, info = self.ids_stream.getnext(return_tensor=False)
        elif isinstance(for_image, tuple):
            frame, info = for_image
        else:
            raise ValueError("for_image has to be a tuple of type (frame, info)")
        query_time = info['timestamp'].total_seconds()
        self.wait_for_n_poses(self.mocap_stream.buffer_size // 2) # Ensure the buffer has enough poses to match
        current_buffer = self.mocap_stream.get_current_buffer() # Get the current buffer of mocap data

        # Filter the buffer for valid mocap data based on tracking validity and marker error threshold
        marker_error_threshold = 0.001
        interest_buffer = []
        for data in current_buffer:
            if data['tracking_valid'] and data['mean_error'] < marker_error_threshold:
                interest_buffer.append(data)

        # Ensure at least 2 poses before and 2 poses after the query_time
        times = [data['timestamp'].total_seconds() for data in interest_buffer]
        before_count = sum(t < query_time for t in times)
        after_count = sum(t > query_time for t in times)
        if before_count < 2 or after_count < 2:
            info = {'is_valid' : False, 'pose' : None, 'pose_velocity' : None}
            if return_tensor:
                frame = torch.from_numpy(frame).permute(2, 0, 1).cuda().float() / 255.0
            return frame, info
        
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
        interpolated_position = np.array([s(query_time) for s in pos_splines])
        interpolated_rotation = rot_spline(query_time).as_quat(scalar_first=True)

        # Query velocities at the camera timestamp
        interpolated_linear_velocity_vec = np.array([s(query_time, 1) for s in pos_splines])
        interpolated_lateral_velocity = np.linalg.norm(interpolated_linear_velocity_vec)

        interpolated_angular_velocity_vec = rot_spline(query_time, 1)
        interpolated_angular_velocity = np.linalg.norm(interpolated_angular_velocity_vec)

        # Create return dict
        pose = {'pos' : interpolated_position, 'rot' : interpolated_rotation}
        pose_velocity = {'pos' : interpolated_lateral_velocity, 'rot' : interpolated_angular_velocity}
        info = {'is_valid' : True, 'pose' : pose, 'pose_velocity' : pose_velocity}
        if return_tensor:
            frame = torch.from_numpy(frame).permute(2, 0, 1).cuda().float() / 255.0

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

        return frame, info

    def stop(self):
        self.running = False
        self.resync_thread.join()