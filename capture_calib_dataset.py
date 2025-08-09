"""
This script captures a dataset for camera- and hand-eye calibration using an IDS camera and a MoCap system.
Therefore it lets the user capture images of a checkerboard pattern, which are then saved together with the corresponding MoCap poses.

This script creates a new directory in /data/ for the calibration data, which contains the following files:
    - images/ : Contains the captured images of the checkerboard pattern
    - mocap_poses.txt : Contains the captured MoCap poses (poses of the rig with respect to the MoCap base) --> T_tool2base

Flags:
    --name : Name of the dataset, default is "calibration_<timestamp>"

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""
import argparse
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
from streams.stream_matcher import StreamMatcher
from capture.capture_dataset import capture_dataset
from datetime import datetime

# Get dataset name from command line argument or use current timestamp as default
parser = argparse.ArgumentParser(description="Calibration Dataset Capture Script")
parser.add_argument("--name", type=str, default=None, help="Name of the dataset")
args = parser.parse_args()

if args.name:
    dataset_path = f"./data/{args.name}"
else:
    dataset_path = f"./data/calibration_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Initialize streams
cam_stream = IDSStream(frame_rate='max', 
                        exposure_time='auto', 
                        white_balance='auto',
                        gain='auto',
                        gamma=1.0)

mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                            server_ip="172.22.147.182", 
                            rigid_body_id=2, # 1 for calibration wand, 2 for camera rig
                            buffer_size=15)

matcher = StreamMatcher(cam_stream, mocap_stream, resync_interval=10, calib_base_path=None, calib_run=None) # No calib because for hand-eye calibration we need the raw MoCap poses
matcher.start_timing()

# Capture dataset
capture_dataset(matcher, dataset_path, mode='auto')

# Stop streams
matcher.stop()
cam_stream.stop()
mocap_stream.stop()

print(f"Dataset captured and saved to {dataset_path}")
print("You can now use the captured data for camera- and hand-eye calibration using the script 'perform_calibration.py'.")
print("Make now sure to check the captured images and delete any that are not suitable for calibration (e.g., blurry images).")
print("The script 'perform_calib.py' will then automatically delete all poses that do not have a corresponding image.")