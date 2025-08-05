import numpy as np
import cv2
from hec.general_utils import read_poses
from scipy.spatial.transform import Rotation as R

# Let's define the following abbreviations for this script: ------------------------------
# CCS - Camera Coordinate System
# TCS - Target Coordinate System (defined by checkerboard)
# BCS - Base Coordinate System (defined by ground plate of mocap system)
# GCS - Gripper Coordinate System (defined by rigid body definition in motive)

# Read poses -----------------------------------------------------------------------------
R_target2cam, t_target2cam = read_poses("./data/hec_checkerboard/checkerboard_poses.txt") # Position of TCS with respect to CCS <-> performs change of basis from TCS to CCS
R_gripper2base, t_gripper2base = read_poses("./data/hec_checkerboard/mocap_poses.txt") # Position of GCS with respect to BCS <-> performs change of basis from GCS to BCS

# Perform OpenCV-based linear Robot-World-Hand-Eye-Calibration ------------------------------
R_world2cam = R_target2cam
t_world2cam = t_target2cam
R_base2gripper = [np.linalg.inv(rot) for rot in R_gripper2base]
t_base2gripper = [-np.linalg.inv(rot) @ trans for trans, rot in zip(t_gripper2base, R_gripper2base)]
R_base2world, t_base2world, R_gripper2cam, t_gripper2cam   = cv2.calibrateRobotWorldHandEye(R_world2cam, t_world2cam, R_base2gripper, t_base2gripper, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)

# Check if T_C^W @ T_G^C @ T_B^G @ T_W^B = I holds true for the first pose
T_C2W = np.eye(4)
T_C2W[:3, :3] = R_world2cam[0]
T_C2W[:3, 3] = t_world2cam[0].flatten()
T_C2W = np.linalg.inv(T_C2W) 
T_G2C = np.eye(4)
T_G2C[:3, :3] = R_gripper2cam
T_G2C[:3, 3] = t_gripper2cam.flatten()
T_B2G = np.eye(4)
T_B2G[:3, :3] = R_base2gripper[0]
T_B2G[:3, 3] = t_base2gripper[0].flatten()
T_W2B = np.eye(4)
T_W2B[:3, :3] = R_base2world
T_W2B[:3, 3] = t_base2world.flatten()
T_W2B = np.linalg.inv(T_W2B)  # Position of BCS with respect to WCS <-> performs change of basis from BCS to WCS
check = T_C2W @ T_G2C @ T_B2G @ T_W2B

print(check)

"""
# Perform OpenCV-based linear Hand-Eye-Calibration ---------------------------------------
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_PARK)
T_cam2gripper = np.eye(4)
T_cam2gripper[:3, :3] = R_cam2gripper
T_cam2gripper[:3, 3] = t_cam2gripper.flatten() # Hand-Eye-Pose: Position of CCS with respect to GCS <-> performs change of basis from CCS to GCS
T_gripper2cam = np.linalg.inv(T_cam2gripper) # Hand-Eye-Pose: Position of GCS with respect to CCS <-> performs change of basis from GCS to CCS

# Save Hand-Eye-Pose ---------------------------------------------------------------------
np.savetxt("./data/hec_checkerboard/hand_eye_pose.txt", T_gripper2cam, fmt='%.6f')
# To retrieve the desired Base-to-Camera Transformation Matrices, perform:
# T_base2cam = T_gripper2cam @ T_base2gripper -> Position of BCS with respect to CCS <-> performs change of basis from BCS to CCS
"""