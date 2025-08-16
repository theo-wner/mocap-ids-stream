"""
This script takes a calibration dataset captured with the script `capture_calib_dataset.py` and performs camera and hand-eye calibration.

This script takes in a calibration directory in /data/ and adds the following files:
    - checkerboard_poses.txt : Contains the poses of the checkerboard pattern with respect to the camera coordinate system --> T_world2cam
    - intrinsics.txt : Contains the camera intrinsics
    - hand_eye_pose.txt : Contains the Hand-Eye-Pose (pose of the tool with respect to the cam) --> T_tool2cam

The created directory can then be passed as an argument to the StreamMatcher class, which is then able to correct the MoCap poses by the Hand-Eye-Pose and the camera intrinsics.

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""
import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from calibration.utils import filter_poses
from calibration.camera_calibration import perform_camera_calibration
from calibration.hand_eye_calibration import perform_hand_eye_calibration, apply_hand_eye_transform

# Get dataset name from command line argument or use the one with the latest timestamp as default
parser = argparse.ArgumentParser(description="Calibration Script")
parser.add_argument("--name", type=str, default=None, help="Name of the dataset")
args = parser.parse_args()

if args.name:
    dataset_path = f"./data/{args.name}"
else:
    # Use the latest dataset by timestamp if no name is provided
    dataset_path = sorted([d for d in os.listdir('./data/') if d.startswith('calibration_')], reverse=True)[0]
    dataset_path = os.path.join('./data/', dataset_path)
print(f"Using dataset path: {dataset_path}")

# Ensure that all mocap poses have a corresponding image
filter_poses(dataset_path)

# Perform camera calibration
print("Performing camera calibration...")
perform_camera_calibration(dataset_path)

# Perform hand-eye calibration
print("Performing hand-eye calibration...")
perform_hand_eye_calibration(dataset_path)

# Apply the Hand-Eye-Pose to the MoCap poses to recieve
print("Applying Hand-Eye-Pose to MoCap poses...")
apply_hand_eye_transform(dataset_path)
