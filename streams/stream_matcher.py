"""
Module for matching data and handling time sync of an IDS Camera Stream and a OptiTrack MoCap Stream

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""

import threading
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import cv2
import torch
import os
from scipy.spatial.transform import Rotation as R

class StreamMatcher:
    """
    A class to handle Streams from both an IDS Camera and a OptiTrack MoCap System
    """
    def __init__(self, ids_stream, mocap_stream, rb_id, calib_path=None, undistort=False, downsampling=1):
        self.ids_stream = ids_stream
        self.mocap_stream = mocap_stream
        self.rb_id = rb_id
        self.calib_path = calib_path
        self.undistort = undistort
        self.downsampling = downsampling
        self.latency_diff = 0.040 # 40 ms

        # Set calibration if provided
        if self.calib_path is not None:
            if self.calib_path == "latest":
                calib_dir = os.path.join(".", "data", "calibrations")
                calib_run = sorted([d for d in os.listdir(calib_dir) if d.startswith('calibration_')], reverse=True)[0]
                self.calib_path = os.path.join(calib_dir, calib_run)
                print(f"Using latest calibration directory: {self.calib_path}")
            else:
                print(f"Using specified calibration directory: {self.calib_path}")

            # Intrinsics
            self.intrinsics = {}
            with open(os.path.join(self.calib_path, "sparse", "0", "cameras.txt"), "r") as f:
                for line in f:
                    if line.startswith("1 PINHOLE"):
                        line = line.strip().split(" ")
                        fx = float(line[4])
                        fy = float(line[5])
                        self.intrinsics["simple_pinhole"] = {"f" : (fx + fy) / 2}
                        continue

                    if line.startswith("2 OPENCV"):
                        line = line.strip().split(" ")
                        fx = float(line[4])
                        fy = float(line[5])
                        cx = float(line[6])
                        cy = float(line[7])
                        k1 = float(line[8])
                        k2 = float(line[9])
                        p1 = float(line[10])
                        p2 = float(line[11])
                        camera_matrix = np.array([[fx, 0, cx],
                                                  [0, fy, cy],
                                                  [0, 0, 1]])
                        distortion = np.array([k1, k2, p1, p2])
                        self.intrinsics["opencv"] = {"camera_matrix" : camera_matrix,
                                                     "distortion" : distortion}

            # Hand-Eye Calibration
            self.hand_eye_pose = np.loadtxt(os.path.join(self.calib_path, "sparse", "0", "T_tool2cam_opt.txt"))

        # Set calibration to None if not provided
        else:
            self.intrinsics = None
            self.hand_eye_pose = None
            print("No calibration directory provided. Intrinsics and Hand-Eye Calibration will not be applied.")

    def __len__(self):
        # Arbitrary large number as we don't know the length of a stream
        return 100_000_000  
    
    def get_image_size(self):
        height, width = self.ids_stream.get_image_size()
        height //= self.downsampling
        width //= self.downsampling
        return (height, width) 
    
    def get_original_image_size(self):
        height, width = self.ids_stream.get_image_size()
        return (height, width) 
    
    def get_calib_path(self):
        return None
    
    def get_focal(self):
        return self.intrinsics["simple_pinhole"]["f"] if self.intrinsics else None

    def getnext(self, return_tensor=True):
        # Get next frame and current mocap buffer
        frame, timestamp = self.ids_stream.getnext()
        corrected_timestamp = timestamp - self.latency_diff

        pose_buffer = self.mocap_stream.get_current_buffer()
        times = []
        positions = []
        rotations = []
        errors = []

        for pose in pose_buffer:
            if pose["rigid_bodies"][self.rb_id]["tracking_valid"]:
                times.append(pose["timestamp"])
                positions.append(pose["rigid_bodies"][self.rb_id]["pos"])
                rotations.append(pose["rigid_bodies"][self.rb_id]["rot"])
                errors.append(pose["rigid_bodies"][self.rb_id]["mean_error"])

        # Return None if buffer is too sparse
        if len(times) < 3:
            return frame, None
        
        # Sort by time
        sorted_indices = sorted(range(len(times)), key=lambda k: times[k])
        times = [times[i] for i in sorted_indices]
        positions = [positions[i] for i in sorted_indices]
        rotations = [rotations[i] for i in sorted_indices]
        errors = [errors[i] for i in sorted_indices]
        
        # Get best fitting pose
        time_diffs = np.abs(np.array(times) - corrected_timestamp)
        best_idx = np.argmin(time_diffs)

        # Return None if index is 0 (best pose is likely too old for buffer) or index is last one (best pose is likely too new for buffer)
        if best_idx == 0 or best_idx == len(times) - 1:
            return frame, None
        
        # Prepare return values
        m_pos = np.mean(np.std(positions, axis=0)) * 1000
        m_rot = np.mean(np.std(rotations, axis=0)) * 1000
        focal = self.intrinsics["simple_pinhole"]["f"] if self.intrinsics else None
        camera_matrix = self.intrinsics["opencv"]["camera_matrix"] if self.intrinsics else None
        distortion = self.intrinsics["opencv"]["distortion"] if self.intrinsics else None
        T_tool2base = np.eye(4)
        T_tool2base[0:3, 0:3] = R.from_quat(rotations[best_idx], scalar_first=False).as_matrix()
        T_tool2base[0:3, 3] = positions[best_idx]

        # Apply Hand-Eye-Pose if available
        if self.hand_eye_pose is None:
            return_transform = T_tool2base
            return_pos = positions[best_idx]
            return_rot = rotations[best_idx]

        elif self.hand_eye_pose is not None:
            T_base2tool = np.linalg.inv(T_tool2base)
            T_base2cam = self.hand_eye_pose @ T_base2tool
            return_transform = T_base2cam
            return_pos = T_base2cam[0:3, 3]
            return_rot = R.from_matrix(T_base2cam[0:3, 0:3]).as_quat(scalar_first=False)

        if self.undistort:
            h, w = self.get_original_image_size()
            fx = camera_matrix[0][0]
            fy = camera_matrix[1][1]
            cx = w / 2
            cy = h / 2
            new_camera_matrix = np.array([[fx, 0, cx],
                                            [0, fy, cy],
                                            [0, 0, 1]])
            frame = cv2.undistort(frame, camera_matrix, distortion, None, new_camera_matrix)
            focal = (fx + fy) / 2

        if self.downsampling > 1:
            frame = cv2.resize(frame, self.get_image_size(), interpolation=cv2.INTER_AREA)
            focal /= self.downsampling

        if return_tensor:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).cuda().float() / 255.0
            return_transform = torch.from_numpy(return_transform).cuda().float()
            focal = torch.tensor(focal).cuda().float().unsqueeze(0)

        info = {"pos" : return_pos,
                "rot" : return_rot,
                "m_pos" : m_pos,
                "m_rot" : m_rot,
                "mean_error" : errors[best_idx],
                "timestamp" : times[best_idx],
                "time_diff" : time_diffs[best_idx],
                "camera_matrix" : camera_matrix,
                "distortion" : distortion,
                "Rt" : return_transform, # for on-the-fly-nvs
                "focal" : focal, # for on-the-fly-nvs
                "is_test" : False,} # for on-the-fly-nvs
        
        return frame, info

    def get_time_diff(self):
        """
        Runs the sync_event-method on both streams (IDS in main thread due to GUI, mocap in separate thread)
        and returns their latency difference
        """
        stop_event = threading.Event()
        mocap_results = {}

        # Run MoCap sync in a separate thread (no GUI in this thread)
        def mocap_runner():
            mocap_results.update(self.mocap_stream.sync_event(stop_event=stop_event))

        mocap_thread = threading.Thread(target=mocap_runner)
        mocap_thread.start()

        # Run camera sync in main thread
        ids_results = self.ids_stream.sync_event()

        # Stop MoCap acquisition
        stop_event.set()
        mocap_thread.join()

        # Extract times
        ids_time = ids_results["interest_time"]
        mocap_time = mocap_results["interest_time"]
        t0_ids = ids_results["original_times"][0]
        t0_mocap = mocap_results["original_times"][0]
        time_diff = (ids_time - mocap_time) * 1000  # ms

        # Plot results
        # --- LaTeX-style / mathtext ---
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 15,
            'mathtext.fontset': 'stix',
            'font.family': 'STIXGeneral',
            'axes.formatter.useoffset': False,  # no scientific notation
        })

        # --- Compute common x-axis limits ---
        xmin = min(
            ids_results["original_times"].min(),
            ids_results["interp_times"].min(),
            ids_time.min(),
            mocap_results["original_times"].min(),
            mocap_results["interp_times"].min(),
            mocap_time.min(),
        )
        xmax = max(
            ids_results["original_times"].max(),
            ids_results["interp_times"].max(),
            ids_time.max(),
            mocap_results["original_times"].max(),
            mocap_results["interp_times"].max(),
            mocap_time.max(),
        )

        # --- Shift all times to start at zero (relative) ---
        ids_times_rel = ids_results["original_times"] - xmin
        ids_interp_rel = ids_results["interp_times"] - xmin
        ids_interest_rel = ids_time - xmin

        mocap_times_rel = mocap_results["original_times"] - xmin
        mocap_interp_rel = mocap_results["interp_times"] - xmin
        mocap_interest_rel = mocap_time - xmin

        # --- Create subplots ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- IDS plot ---
        ax1.plot(ids_times_rel, ids_results["original_rows"], 'o', label="Original Points", markersize=4)
        ax1.plot(ids_interp_rel, ids_results["interp_rows"], '-', label="Cubic Spline")
        ax1.plot(ids_interest_rel, ids_results["interest_row"], 'rx', markersize=10, label="Minimum Row")
        ax1.set_xlim(0, xmax - xmin)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Row (px)")
        ax1.invert_yaxis()
        ax1.grid(True)
        ax1.legend(loc='lower left')  # locked legend
        ax1.set_title("Cam Sync Event")

        # IDS sampling info
        ids_interval = ids_results["original_times"][-1] - ids_results["original_times"][0]
        ids_samples = len(ids_results["original_times"])
        ids_fps = ids_samples / ids_interval if ids_interval > 0 else 0
        ids_dt_ms = (ids_interval / (ids_samples - 1) * 1000) if ids_samples > 1 else 0
        ax1.text(0.95, 0.05, f"Sampling Frequency: {ids_fps:.2f} Hz\nSampling Interval: {ids_dt_ms:.2f} ms",
                transform=ax1.transAxes, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

        # --- MoCap plot ---
        ax2.plot(mocap_times_rel, mocap_results["original_ys"], 'o', label="Original Points", markersize=4)
        ax2.plot(mocap_interp_rel, mocap_results["interp_ys"], '-', label="Cubic Spline")
        ax2.plot(mocap_interest_rel, mocap_results["interest_y"], 'rx', markersize=10, label="Maximum Y")
        ax2.set_xlim(0, xmax - xmin)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Y (m)")
        ax2.grid(True)
        ax2.legend(loc='lower left')  # locked legend
        ax2.set_title("MoCap Sync Event")

        # MoCap sampling info
        mocap_interval = mocap_results["original_times"][-1] - mocap_results["original_times"][0]
        mocap_samples = len(mocap_results["original_times"])
        mocap_fps = mocap_samples / mocap_interval if mocap_interval > 0 else 0
        mocap_dt_ms = (mocap_interval / (mocap_samples - 1) * 1000) if mocap_samples > 1 else 0
        ax2.text(0.95, 0.05, f"Sampling Frequency: {mocap_fps:.2f} Hz\nSampling Interval: {mocap_dt_ms:.2f} ms",
                transform=ax2.transAxes, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

        # --- Format x-axis to ms ---
        for ax in (ax1, ax2):
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*1000:.0f}"))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # --- Format y-axis of MoCap to 3 decimals ---
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}"))

        # --- Figure title ---
        plt.suptitle(f"Time difference (IDS - MoCap): {time_diff:.2f} ms\nPress 'k' to keep, 'r' to reject")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Wait for user decision
        decision = {"keep": None}
        def on_key(event):
            if event.key == "k":
                decision["keep"] = True
                plt.close(fig)
            elif event.key == "r":
                decision["keep"] = False
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        return time_diff if decision["keep"] else None
