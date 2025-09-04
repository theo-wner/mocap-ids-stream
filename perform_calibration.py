"""
This script takes a calibration dataset captured with the script `capture_calib_dataset.py` and performs camera and hand-eye calibration.

This script takes in a calibration directory and adds the following files to it:
    - sparse/0/images_checkerboard.txt : Contains the poses of the checkerboard pattern with respect to the camera coordinate system --> T_world2cam
    - sparse/0/cameras.txt : Contains the camera intrinsics
    - sparse/0/hand_eye_pose.txt : Contains the Hand-Eye-Pose (pose of the tool with respect to the cam) --> T_tool2cam
    - sparse/0/images.txt : Contains the MoCap poses corrected by the Hand-Eye-Transform (poses of the MoCap base with respect to the CCS) --> T_base2cam

The path to this calibration directory can then be passed as an argument to the StreamMatcher class, which is then able to correct the MoCap poses by the Hand-Eye-Pose and the camera intrinsics.

Flags:
    --calib_path : Path to the previously captured calibration dataset. 
                    Either pass a custom path or 'latest' for the latest timestamped calibration dataset in './data/calibrations'.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""
import argparse
import os
from scipy.spatial.transform import Rotation as R
from calibration.utils import filter_poses, apply_hand_eye_transform
from calibration.camera_calibration import perform_camera_calibration
from calibration.hand_eye_calibration import perform_hand_eye_calibration, perform_robot_world_hand_eye_calibration, refine_hand_eye_pose

# Get dataset name from command line argument or use the one with the latest timestamp as default
parser = argparse.ArgumentParser(description="Calibration Script")
parser.add_argument("--calib_path", type=str, default=None, required=True, help="Path to the previously captured calibration dataset. Either pass a custom path or 'latest' for the latest timestamped calibration dataset in './data/calibrations'.")
args = parser.parse_args()

if args.calib_path == "latest":
    calib_dir = os.path.join(".", "data", "calibrations")
    calib_run = sorted([d for d in os.listdir(calib_dir) if d.startswith('calibration_')], reverse=True)[0]
    calib_path = os.path.join(calib_dir, calib_run)
else:
    calib_path = f"{args.calib_path}"
print(f"Using calibration path: {calib_path}")

# Ensure that all mocap poses have a corresponding image
filter_poses(calib_path)

# Perform camera calibration
# Define Chessboard
chessboard = {'num_corners_down' : 24,
                'num_corners_right' : 17,
                'origin_marker_pos_down' : 11,
                'origin_marker_pos_right' : 7,
                'square_size' : 30}
print("Performing camera calibration...")
perform_camera_calibration(calib_path, chessboard)

# Perform hand-eye calibration
print("Performing hand-eye calibration...")
perform_robot_world_hand_eye_calibration(calib_path)

# Refine hand-eye calibration
print("Refining Hand-Eye-Calibration...")
refine_hand_eye_pose(calib_path, opt_mocap_poses=True, opt_intrinsics=True)

