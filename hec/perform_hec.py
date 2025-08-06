import numpy as np
import cv2
from hec.general_utils import read_poses
from scipy.spatial.transform import Rotation as R

# Let's define the following abbreviations for this script: ------------------------------
# CCS - Camera Coordinate System
# WCS - World Coordinate System (defined by checkerboard)
# BCS - Base Coordinate System (defined by ground plate of mocap system)
# TCS - Tool Coordinate System (defined by rigid body definition in motive)

# Read poses -----------------------------------------------------------------------------
R_world2cam, t_world2cam = read_poses("./data/hec_checkerboard/checkerboard_poses.txt") # Position of WCS with respect to CCS <-> performs change of basis from WCS to CCS
R_tool2base, t_tool2base = read_poses("./data/hec_checkerboard/mocap_poses.txt") # Position of TCS with respect to BCS <-> performs change of basis from TCS to BCS

# Perform OpenCV-based linear Hand-Eye-Calibration ---------------------------------------
R_cam2tool, t_cam2tool = cv2.calibrateHandEye(R_tool2base, t_tool2base, R_world2cam, t_world2cam, method=cv2.CALIB_HAND_EYE_PARK)
T_cam2tool = np.eye(4)
T_cam2tool[:3, :3] = R_cam2tool
T_cam2tool[:3, 3] = t_cam2tool.flatten() # Hand-Eye-Pose: Position of CCS with respect to TCS <-> performs change of basis from CCS to TCS
T_tool2cam = np.linalg.inv(T_cam2tool) # Hand-Eye-Pose: Position of TCS with respect to CCS <-> performs change of basis from TCS to CCS

# Save Hand-Eye-Pose ---------------------------------------------------------------------
np.savetxt("./data/hec_checkerboard/hand_eye_pose.txt", T_tool2cam, fmt='%.6f')
# To retrieve the desired Base-to-Camera Transformation Matrices, perform:
# T_base2cam = T_tool2cam @ T_base2tool -> Position of BCS with respect to CCS <-> performs change of basis from BCS to CCS