"""
This script captures a dataset for camera- and hand-eye calibration using an IDS camera and a MoCap system.
Therefore it lets the user capture images of a checkerboard pattern, which are then saved together with the corresponding MoCap poses.

This script creates a new directory for the calibration data, which contains the following files:
    - images/ : Contains the captured images of the checkerboard pattern
    - sparse/0/images_mocap.txt : Contains the captured MoCap poses (poses of the rig with respect to the MoCap base) --> T_tool2base
    - sparse/0/points3D.txt : Empty placeholder file

Flags:
    --calib_path : Path to the calibration dataset to be captured. Gets created in this script. 
                    Either pass a custom path or 'default' for './data/calibrations/calibration_<timestamp>'

Author:
    Theodor Kapler <theodor.kapler@student.kit.edu>
"""
import argparse
import os
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
from streams.stream_matcher import StreamMatcher
from capture.capture_dataset import capture_dataset
from datetime import datetime

# Get dataset name from command line argument or use current timestamp as default
parser = argparse.ArgumentParser(description="Calibration Dataset Capture Script")
parser.add_argument("--calib_path", type=str, default=None, required=True, help="Path to the calibration dataset to be captured. Gets created in this script. Either pass a custom path or 'default' for './data/calibrations/calibration_<timestamp>'")
args = parser.parse_args()

if args.calib_path == "default":
    calib_path = os.path.join(".", "data", "calibrations", f"calibration_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
else:
    calib_path = f"{args.calib_path}"
print(f"Using calibration path: {calib_path}")

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

matcher = StreamMatcher(cam_stream, mocap_stream, resync_interval=1, calib_path=None) # No calib because for hand-eye calibration we need the raw MoCap poses
matcher.start_timing()

# Capture dataset
capture_dataset(matcher, calib_path, mode='auto')

# Stop streams
matcher.stop()
cam_stream.stop()
mocap_stream.stop()

print(f"Dataset captured and saved to {calib_path}")
print("You can now use the captured data for camera- and hand-eye calibration using the script 'perform_calibration.py'.")
print("Make now sure to check the captured images and delete any that are not suitable for calibration (e.g., blurry images).")
print("The script 'perform_calibration.py' will then automatically delete all poses that do not have a corresponding image.")