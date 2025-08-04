import numpy as np
import cv2
from hec.general_utils import filter_poses, read_poses
from hec.colmap_utils import convert_to_tum
from scipy.spatial.transform import Rotation as R

# Let's define the following abbreviations for this script: ------------------------------
# CCS - Camera Coordinate System
# TCS - Target Coordinate System (defined by checkerboard)
# BCS - Base Coordinate System (defined by ground plate of mocap system)
# GCS - Gripper Coordinate System (defined by rigid body definition in motive)

# Read poses -----------------------------------------------------------------------------
checkerboard_poses = read_poses("./data/hec_checkerboard/checkerboard_poses.txt") # Position of TCS with respect to CCS <-> performs change of basis from TCS to CCS
mocap_poses = read_poses("./data/hec_checkerboard/mocap_poses.txt") # Position of GCS with respect to BCS <-> performs change of basis from GCS to BCS

R_target2cam, t_target2cam, R_gripper2base, t_gripper2base = [], [], [], []
for key in checkerboard_poses.keys():
    R_target2cam.append(checkerboard_poses[key][:3, :3])
    t_target2cam.append(checkerboard_poses[key][:3, 3])
    R_gripper2base.append(mocap_poses[key][:3, :3])
    t_gripper2base.append(mocap_poses[key][:3, 3])

# Perform OpenCV-based linear Hand-Eye-Calibration ---------------------------------------
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_target2cam, t_target2cam, R_gripper2base, t_gripper2base, method=cv2.CALIB_HAND_EYE_PARK)
T_cam2gripper = np.eye(4)
T_cam2gripper[:3, :3] = R_cam2gripper
T_cam2gripper[:3, 3] = t_cam2gripper.flatten() # Hand-Eye-Pose: Position of CCS with respect to GCS <-> performs change of basis from CCS to GCS
T_gripper2cam = np.linalg.inv(T_cam2gripper) # Hand-Eye-Pose: Position of GCS with respect to CCS <-> performs change of basis from GCS to CCS

# Apply computed Hand-Eye-Pose -----------------------------------------------------------
transformed_mocap_poses = {}
for key, T_gripper2base in mocap_poses.items():
    T_base2gripper = np.linalg.inv(T_gripper2base)
    #transformed_mocap_poses[key] = T_base2gripper @ T_gripper2cam
    transformed_mocap_poses[key] = T_gripper2base @ T_cam2gripper

with open("./data/hec_checkerboard/transformed_mocap_poses.txt", "w") as f:
    f.write("IMAGE_ID QW QX QY QZ TX TY TZ IMAGE_NAME\n")
    for key, T in transformed_mocap_poses.items():
        quat = R.from_matrix(T[:3, :3]).as_quat()
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        tx, ty, tz = T[:3, 3]
        image_name = key 
        f.write(f"{key} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {image_name}\n")

with open("./data/hec_checkerboard/inverted_checkerboard_poses.txt", "w") as f:
    f.write("IMAGE_ID QW QX QY QZ TX TY TZ IMAGE_NAME\n")
    for key, T in checkerboard_poses.items():
        T = np.linalg.inv(T)
        quat = R.from_matrix(T[:3, :3]).as_quat()
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        tx, ty, tz = T[:3, 3]
        image_name = key 
        f.write(f"{key} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {image_name}\n")

# Convert poses to TUM-format for evaluation with evo ------------------------------------
convert_to_tum("./data/hec_checkerboard/mocap_poses.txt", "./data/hec_checkerboard/mocap_poses_tum.txt")
convert_to_tum("./data/hec_checkerboard/checkerboard_poses.txt", "./data/hec_checkerboard/checkerboard_poses_tum.txt")
convert_to_tum("./data/hec_checkerboard/transformed_mocap_poses.txt", "./data/hec_checkerboard/transformed_mocap_poses_tum.txt")
convert_to_tum("./data/hec_checkerboard/inverted_checkerboard_poses.txt", "./data/hec_checkerboard/inverted_checkerboard_poses_tum.txt")