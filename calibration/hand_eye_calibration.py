import numpy as np
import cv2
import os
import pickle
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

def perform_robot_world_hand_eye_calibration(dataset_path):
    """
    This function performs robot-world-hand-eye-calibration using the "robot"-poses retrieved by MoCap and the camera poses retrieved by camera calibration.
    It then saves the calculated Hand-Eye-Pose as well as the Robot-World pose to a file.

    Args:
        dataset_path (str): The path where the dataset is saved, which contains the checkerboard poses the MoCap poses,
                            and a file 'checkerboard_points.pkl' to store the object points and image points for computation of the reprojection error.
    """
    # Read poses -----------------------------------------------------------------------------
    R_world2cam, t_world2cam = read_poses(os.path.join(dataset_path, "sparse", "0", "images_checkerboard.txt"))
    R_tool2base, t_tool2base = read_poses(os.path.join(dataset_path, "sparse", "0", "images_mocap.txt"))

    # Invert MoCap poses ---------------------------------------------------------------------
    R_base2tool = [np.linalg.inv(entry) for entry in R_tool2base]
    t_base2tool = [-entry_r @ entry_t for entry_r, entry_t in zip(R_base2tool, t_tool2base)]

    # Perform OpenCV-based linear Hand-Eye-Robot-World-Calibration ---------------------------
    R_base2world, t_base2world, R_tool2cam, t_tool2cam = cv2.calibrateRobotWorldHandEye(R_world2cam, t_world2cam, R_base2tool, t_base2tool, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)

    # Calculate reprojection error ----------------------------------------------------------
    # Load intrinsics
    with open(os.path.join(dataset_path, "sparse", "0", "cameras.txt"), "r") as f:
        for line in f:
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

    # Load object points and image points
    with open(os.path.join(dataset_path, "sparse", "0", "checkerboard_points.pkl"), "rb") as f:
        points = pickle.load(f)

    objpoints = points["objpoints"]
    imgpoints = points["imgpoints"]

    # Transform Object Points from WCS to CCS
    T_tool2cam = np.eye(4)
    T_tool2cam[:3, :3] = R_tool2cam
    T_tool2cam[:3, 3] = t_tool2cam.flatten()

    T_base2world = np.eye(4)
    T_base2world[:3, :3] = R_base2world
    T_base2world[:3, 3] = t_base2world.flatten()

    transformed_objpoints = []

    for i in range(len(objpoints)): # Loop over each image
        T_world2cam = np.eye(4)
        T_world2cam[:3, :3] = R_world2cam[i]
        T_world2cam[:3, 3] = t_world2cam[i].flatten()

        T_base2tool = np.eye(4)
        T_base2tool[:3, :3] = R_base2tool[i]
        T_base2tool[:3, 3] = t_base2tool[i].flatten()

        objp = np.hstack((objpoints[i], np.ones((objpoints[i].shape[0], 1)))) # Create homogenous coordinates
        transformed_objp = T_tool2cam @ T_base2tool @ np.linalg.inv(T_base2world) @ objp.T # Transform Object Points from WCS to CCS
        transformed_objp = transformed_objp.T[:, :3] # Remove homogenous component
        transformed_objpoints.append(transformed_objp) 

        # Check loop closure:
        loop = T_tool2cam @ T_base2tool @ np.linalg.inv(T_base2world) @ np.linalg.inv(T_world2cam)
        is_identity = np.allclose(loop, np.eye(4), atol=1e-2)
        if not is_identity:
            print(f"Loop Closure condition not satisfied for image {i}")

    # Project transformed object points into image plane and calculate mean reprojection error
    rvec = np.array([0., 0., 0.]) # transformed_objpoints are already in CCS
    tvec = np.array([0., 0., 0.])
    total_error = 0
    total_points = 0
    for i in range(len(transformed_objpoints)):
        reprojected_points, _ = cv2.projectPoints(transformed_objpoints[i], rvec, tvec, camera_matrix, distortion)
        total_error += np.sum(np.abs(imgpoints[i] - reprojected_points)**2)
        total_points += len(transformed_objpoints[i])
    mean_error = np.sqrt(total_error / total_points)
    print(f"[✓] Hand-Eye-Calibration completed with reprojection error: {mean_error:.4f} px.")

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
