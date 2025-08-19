"""
This script captures a dataset for 3D recontruction using an IDS camera and a MoCap system.
It captures and saves images together with the corresponding MoCap poses of the camera rig and applies a Hand-Eye-Transform to them.

This script takes in a calibration directory and creates a new directory for the captured dataset containing the following files:
    - images/ : Contains the captured images
    - sparse/0/images.txt : Contains the MoCap poses corrected by the Hand-Eye-Transform (poses of the MoCap base with respect to the CCS) --> T_base2cam
    - sparse/0/cameras.txt : Contains the camera intrinsics
    - sparse/0/points3D.txt : Empty placeholder file

Flags:
    --calib_path : Path to the previously captured calibration dataset. 
                    Either pass a custom path or 'latest' for the latest timestamped calibration dataset in './data/calibrations'.
    
    --dataset_path : Path to the dataset to be captured. Gets created in this script. 
                    Either pass a custom path or 'default' for './data/dataset_<timestamp>'

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""
import argparse
import os
import shutil
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
from streams.stream_matcher import StreamMatcher
from capture.capture_dataset import capture_dataset
from datetime import datetime

# Get dataset name from command line argument or use current timestamp as default
parser = argparse.ArgumentParser(description="Dataset Capture Script")
parser.add_argument("--calib_path", type=str, default=None, required=True, help="Path to the previously captured calibration dataset. Either pass a custom path or 'latest' for the latest timestamped calibration dataset in './data/calibrations'.")
parser.add_argument("--dataset_path", type=str, default=None, required=True, help="Path to the dataset to be captured. Gets created in this script. Either pass a custom path or 'default' for './data/dataset_<timestamp>'")
args = parser.parse_args()

if args.calib_path == "latest":
    calib_dir = os.path.join(".", "data", "calibrations")
    calib_run = sorted([d for d in os.listdir(calib_dir) if d.startswith('calibration_')], reverse=True)[0]
    calib_path = os.path.join(calib_dir, calib_run)
else:
    calib_path = f"{args.calib_path}"
print(f"Using calibration path: {calib_path}")

if args.dataset_path == "default":
    dataset_dir = os.path.join(".", "data")
    dataset_path = os.path.join(dataset_dir, f"dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
else:
    dataset_path = f"{args.dataset_path}"
print(f"Using dataset path: {dataset_path}")

# Initialize streams
cam_stream = IDSStream(frame_rate='max', 
                        exposure_time=20000, 
                        white_balance='auto',
                        gain='auto',
                        gamma=1.5)

mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                            server_ip="172.22.147.182", 
                            rigid_body_id=2, # 1 for calibration wand, 2 for camera rig
                            buffer_size=15)

matcher = StreamMatcher(cam_stream, mocap_stream, resync_interval=10, calib_path=calib_path)
matcher.start_timing()

# Capture dataset
capture_dataset(matcher, dataset_path, mode='auto')

# Copy cameras.txt from calibration path to dataset path
calib_cameras_path = os.path.join(calib_path, "sparse", "0", "cameras.txt")
dataset_cameras_path = os.path.join(dataset_path, "sparse", "0", "cameras.txt")
shutil.copy2(calib_cameras_path, dataset_cameras_path)

# Stop streams
matcher.stop()
cam_stream.stop()
mocap_stream.stop()

print(f"Dataset captured and saved to {dataset_path}")