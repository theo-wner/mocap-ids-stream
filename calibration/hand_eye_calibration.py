import numpy as np
import cv2
import os
from calibration.utils import read_poses
from scipy.spatial.transform import Rotation as R

def perform_hand_eye_calibration(dataset_path):
    """
    This function performs hand-eye-calibration using the "robot"-poses retrieved by MoCap and the camera poses retrieved by camera calibration.
    It then saves the calculated Hand-Eye-Pose to a file.

    Args:
        dataset_path (str): The path where the dataset is saved, which contains the images of the checkerboard pattern, 
                            the checkerboard poses, and the MoCap poses.
    """
    # Let's define the following abbreviations for this script: ------------------------------
    # CCS - Camera Coordinate System
    # WCS - World Coordinate System (defined by checkerboard)
    # BCS - Base Coordinate System (defined by ground plate of mocap system)
    # TCS - Tool Coordinate System (defined by rigid body definition in motive)

    # Read poses -----------------------------------------------------------------------------
    R_world2cam, t_world2cam = read_poses(os.path.join(dataset_path, "sparse", "0", "images_checkerboard.txt")) # Position of WCS with respect to CCS <-> performs change of basis from WCS to CCS
    R_tool2base, t_tool2base = read_poses(os.path.join(dataset_path, "sparse", "0", "images_mocap.txt")) # Position of TCS with respect to BCS <-> performs change of basis from TCS to BCS

    # Perform OpenCV-based linear Hand-Eye-Calibration ---------------------------------------
    R_cam2tool, t_cam2tool = cv2.calibrateHandEye(R_tool2base, t_tool2base, R_world2cam, t_world2cam, method=cv2.CALIB_HAND_EYE_PARK)
    T_cam2tool = np.eye(4)
    T_cam2tool[:3, :3] = R_cam2tool
    T_cam2tool[:3, 3] = t_cam2tool.flatten() # Hand-Eye-Pose: Position of CCS with respect to TCS <-> performs change of basis from CCS to TCS
    T_tool2cam = np.linalg.inv(T_cam2tool) # Hand-Eye-Pose: Position of TCS with respect to CCS <-> performs change of basis from TCS to CCS

    # Save Hand-Eye-Pose ---------------------------------------------------------------------
    np.savetxt(os.path.join(dataset_path, "sparse", "0", "hand_eye_pose.txt"), T_tool2cam, fmt='%.6f')
    print(f"[✓] Saved Hand-Eye-Pose to {os.path.join(dataset_path, "sparse", "0", "hand_eye_pose.txt")}")
    # To retrieve the desired Base-to-Camera Transformation Matrices, perform:
    # T_base2cam = T_tool2cam @ T_base2tool -> Position of BCS with respect to CCS <-> performs change of basis from BCS to CCS

def apply_hand_eye_transform(dataset_path):
    """
    Applies the Hand-Eye-Transform to the MoCap poses and saves the corrected poses to the images.txt file.

    Args:
        dataset_path (str): The path where the dataset is saved, which contains mocap poses (images_mocap.txt) and the Hand-Eye-Pose (hand_eye_pose.txt).
    """
    hand_eye_pose = np.loadtxt(os.path.join(dataset_path, "sparse", "0", "hand_eye_pose.txt"))

    with open(os.path.join(dataset_path, "sparse", "0", "images_mocap.txt"), "r") as f:
        lines = f.readlines()

    with open(os.path.join(dataset_path, "sparse", "0", "images.txt"), "w") as out_f:
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                out_f.write(line)
                continue

            line = line.strip().split(" ")
            img_id = line[0]
            qw, qx, qy, qz, tx, ty, tz = [float(comp) for comp in line[1:8]]
            cam_id = line[8]
            name = line[9]

            pose = np.eye(4)
            pose[:3, :3] = R.from_quat((qx, qy, qz, qw), scalar_first=False).as_matrix()
            pose[:3, 3] = np.array([tx, ty, tz])

            # Apply Hand-Eye-Transform
            pose = hand_eye_pose @ np.linalg.inv(pose) # Results in position of BCS with respect to CCS <-> performs change of basis from BCS to CCS

            qx, qy, qz, qw = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=False)
            tx, ty, tz = pose[:3, 3]

            out_f.write(f"{img_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} 1 {name}\n")

    print(f"[✓] Applied Hand-Eye-Transform to MoCap poses and saved corrected poses to {os.path.join(dataset_path, "sparse", "0", "images.txt")}")