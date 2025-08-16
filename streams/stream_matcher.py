"""
Module for matching data and handling time sync of an IDS Camera Stream and a OptiTrack MoCap Stream

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""
import numpy as np
import threading
import time
import torch
import os
import cv2
from scipy.spatial.transform import Rotation as R, RotationSpline
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class StreamMatcher():
    """
    A class to handle Streams from both an IDS Camera and a OptiTrack MoCap System
    """
    def __init__(self, ids_stream, mocap_stream, resync_interval, print_on_resync=False, calib_base_path=None, calib_run=None, downsampling=None):
        # Streams
        self.ids_stream = ids_stream
        self.mocap_stream = mocap_stream
        self.resync_interval = resync_interval
        self.print_on_resync = print_on_resync
        self.downsampling = downsampling
        
        # Set calibration if provided
        if calib_base_path is not None:
            if calib_run == 'latest':
                calib_run = sorted([d for d in os.listdir(calib_base_path) if d.startswith('calibration_')], reverse=True)[0]
                calib_dir = os.path.join(calib_base_path, calib_run)
                print(f"Using latest calibration directory: {calib_dir}")
            else:
                calib_dir = os.path.join(calib_base_path, calib_run)
                print(f"Using specified calibration directory: {calib_dir}")

            # Intrinsics
            with open(f"{calib_dir}/sparse/0/cameras.txt", "r") as f:
                for line in f:
                    if line.startswith("1 PINHOLE"):
                        line = line.strip().split(" ")
                        fx = float(line[4])
                        fy = float(line[5])

            self.intrinsics = {"FOCAL" : (fx + fy) / 2}

            if self.downsampling is not None:
                self.intrinsics["FOCAL"] /= downsampling

            # Hand-Eye Calibration
            self.hand_eye_pose = np.loadtxt(f"{calib_dir}/sparse/0/hand_eye_pose.txt")

            print(self.intrinsics)

        # Set calibration to None if not provided
        else:
            self.intrinsics = None
            self.hand_eye_pose = None
            print("No calibration directory provided. Intrinsics and Hand-Eye Calibration will not be applied.")

        # Start resync thread
        self.resync_thread = threading.Thread(target=self.resync_loop, daemon=True)
        self.running = True

    def __len__(self):
        # Arbitrary large number as we don't know the length of a stream
        return 100_000_000  
    
    def get_image_size(self):
        if self.downsampling is None:
            return self.ids_stream.get_image_size()
        else:
            height, width = self.ids_stream.get_image_size()
            return (height // self.downsampling, width // self.downsampling) 
        
    def get_focal(self):
        return self.intrinsics['FOCAL'] if self.intrinsics else None

    def start_timing(self):
        self.ids_stream.start_timing()
        self.mocap_stream.start_timing()
        self.resync_thread.start()
        time.sleep(1)

    def resync_loop(self):
        while self.running:
            time.sleep(self.resync_interval)
            if self.print_on_resync:
                print("Resyncronizing timestamps")
            self.ids_stream.resync_timing()
            self.mocap_stream.resync_timing()
            
    def wait_for_n_poses(self, n):
        """
        Waits until `n` new unique mocap poses have appeared in the buffer.
        Uses object identity (memory address) to detect new pose entries.
        """
        buffer = self.mocap_stream.get_current_buffer()
        seen_ids = {id(pose) for pose in buffer}
        
        while len(seen_ids) < len(buffer) + n:
            buffer = self.mocap_stream.get_current_buffer()
            seen_ids.update(id(pose) for pose in buffer)
            time.sleep(0.001)

    def getnext(self, for_image=None, return_tensor=True, show_plot=False):
        """
        Returns the current frame together with its corresponding pose information.
        Can be directly used as getnext-Function inside the onthefly_nvs repo

        Args:
            - for_image (tuple): Optional, if passed, takes image and info as tuple from the argument, if None, retrieves a new image from ids_stream 
            - show_plot (boolean): Determines wheather to show a plot for pose visulization or not

        Returns:
            - frame (torch.Tensor): The captured frame from the IDS Camera Stream, downsampled if specified
            - info (dict): Dictionary containing the pose info
                - keys:
                    - is_valid (boolean): Shows if the interpolated pose is valid
                    - pose (dict): The interpolated pose by the timestamp of image retrieval
                    - pose_velocitiy (dict): The interpolated pose velocities by the timestamp of image retrieval
                    - Rt (torch.Tensor): The 4x4 transformation matrix of the pose --> needed for onthefly_nvs
                    - focal (torch.Tensor): The focal length of the camera, if available --> needed for onthefly_nvs
                    - is_test (boolean): Indicates if the pose is a test pose --> needed for onthefly_nvs
        """
        # Check if for_image is provided, if not, get the next frame from the IDS stream
        if for_image is None:
            frame, info = self.ids_stream.getnext()
        elif isinstance(for_image, tuple):
            frame, info = for_image
        
        # Downsample the image
        if self.downsampling is not None:
            frame = cv2.resize(frame, (0, 0), fx=1/self.downsampling, fy=1/self.downsampling, interpolation=cv2.INTER_AREA)

        # Convert to torch tensor
        if return_tensor:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).cuda().float() / 255.0

        query_time = info['timestamp'].total_seconds()
        self.wait_for_n_poses(self.mocap_stream.buffer_size // 2) # Ensure the buffer has enough poses to match
        current_buffer = self.mocap_stream.get_current_buffer() # Get the current buffer of mocap data

        # Filter the buffer for valid mocap data based on tracking validity and marker error threshold and sort by timestamp
        marker_error_threshold = 0.001
        interest_buffer = sorted(
            [data for data in current_buffer if data['tracking_valid'] and data['mean_error'] < marker_error_threshold],
            key=lambda d: d['timestamp'].total_seconds()
        )

        # Ensure at least 2 poses before and 2 poses after the query_time
        times = [data['timestamp'].total_seconds() for data in interest_buffer]
        before_count = sum(t < query_time for t in times)
        after_count = sum(t > query_time for t in times)
        if before_count < 2 or after_count < 2:
            info = {'is_valid' : False, 
                    'pose' : None, 
                    'pose_velocity' : None, 
                    'Rt' : None, 
                    'focal' : None,
                    'is_test' : False}
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
        interpolated_rotation = rot_spline(query_time).as_quat(scalar_first=False)

        # Query velocities at the camera timestamp
        interpolated_linear_velocity_vec = np.array([s(query_time, 1) for s in pos_splines])
        interpolated_lateral_velocity = np.linalg.norm(interpolated_linear_velocity_vec)

        interpolated_angular_velocity_vec = rot_spline(query_time, 1)
        interpolated_angular_velocity = np.linalg.norm(interpolated_angular_velocity_vec)

        # Convert the pose to a 4x4 transformation matrix
        interpolated_pose = np.eye(4)
        interpolated_pose[:3, :3] = R.from_quat(interpolated_rotation, scalar_first=False).as_matrix()
        interpolated_pose[:3, 3] = interpolated_position
        
        # Apply the Hand-Eye Calibration if available
        if self.hand_eye_pose is not None:
            # Apply the Hand-Eye Calibration
            interpolated_pose = self.hand_eye_pose @ np.linalg.inv(interpolated_pose) # Results in position of BCS with respect to CCS <-> performs change of basis from BCS to CCS

            # Extract the new position and rotation
            interpolated_position = interpolated_pose[:3, 3]
            interpolated_rotation = R.from_matrix(interpolated_pose[:3, :3]).as_quat(scalar_first=False)

        # Create return dict
        pose = {'pos' : interpolated_position, 'rot' : interpolated_rotation}
        pose_velocity = {'pos' : interpolated_lateral_velocity, 'rot' : interpolated_angular_velocity}
        Rt = interpolated_pose
        focal = self.intrinsics['FOCAL'] if self.intrinsics is not None else None
        if return_tensor:
            Rt = torch.from_numpy(Rt).float().cuda()
            focal = torch.from_numpy(np.array(focal)).float().unsqueeze(0).cuda() if focal is not None else None
        info = {'is_valid' : True, 
                'pose' : pose, 
                'pose_velocity' : pose_velocity,
                'Rt' : Rt,
                'focal' : focal,
                'is_test' : False}

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