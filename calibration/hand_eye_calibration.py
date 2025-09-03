import numpy as np
import cv2
import os
from calibration.utils import read_poses, read_calib_points, read_intrinsics
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

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
    np.savetxt(os.path.join(dataset_path, "sparse", "0", "T_tool2cam.txt"), T_tool2cam, fmt='%.6f')
    print(f"[✓] Saved Hand-Eye-Pose to {os.path.join(dataset_path, "sparse", "0", "T_tool2cam.txt")}")
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

    # Create homogenous transform matrices for saving
    T_tool2cam = np.eye(4)
    T_tool2cam[:3, :3] = R_tool2cam
    T_tool2cam[:3, 3] = t_tool2cam.flatten()
    T_base2world = np.eye(4)
    T_base2world[:3, :3] = R_base2world
    T_base2world[:3, 3] = t_base2world.flatten()

    # Save Hand-Eye-Pose and Base-World-Pose----------------------------------------------------
    np.savetxt(os.path.join(dataset_path, "sparse", "0", "T_tool2cam.txt"), T_tool2cam, fmt='%.6f')
    np.savetxt(os.path.join(dataset_path, "sparse", "0", "T_base2world.txt"), T_base2world, fmt='%.6f')
    print(f"[✓] Saved Hand-Eye-Pose to {os.path.join(dataset_path, "sparse", "0", "T_tool2cam.txt")}")
    print(f"[✓] Saved Base-World-Pose to {os.path.join(dataset_path, "sparse", "0", "T_base2world.txt")}")
    # To retrieve the desired Base-to-Camera Transformation Matrices, perform:
    # T_base2cam = T_tool2cam @ T_base2tool -> Position of BCS with respect to CCS <-> performs change of basis from BCS to CCS

def residuals(params, objpoints, imgpoints, camera_matrix, distortion):
    """
    Defines a residual function which can be optimized with scipy.optimize.least_squares.

    Args:
        params (numpy.ndarray): Parameters to optimize for. Here: 
                                    Hand-Eye-Pose
                                    Base-World-Pose 
                                    All MoCap (base2tool)-Poses
                                All poses are parameterized as euler angles (xyz) and translation vector each                    
        other parameters: Fixed parameters needed for the residual calculation

    Returns:
        residuals (numpy.ndarray): Residual vector with two entries for each point: x and y
    """
    # Create homogenous transformation matrices from Hand-Eye-Pose and Base-World-Pose parameters
    rvec_tool2cam = params[:3]
    tvec_tool2cam = params[3:6]
    rvec_base2world = params[6:9]
    tvec_base2world = params[9:12]

    T_tool2cam = np.eye(4)
    T_tool2cam[:3, :3] = R.from_euler("xyz", rvec_tool2cam, degrees=False).as_matrix()
    T_tool2cam[:3, 3] = tvec_tool2cam

    T_base2world = np.eye(4)
    T_base2world[:3, :3] = R.from_euler("xyz", rvec_base2world, degrees=False).as_matrix()
    T_base2world[:3, 3] = tvec_base2world

    params_base2tool = params[12:]
    T_base2tool = []
    for i in range(len(objpoints)):
        rvec = params_base2tool[i*6:i*6+3]
        tvec = params_base2tool[i*6+3:i*6+6]
        mtx = np.eye(4)
        mtx[:3, :3] = R.from_euler("xyz", rvec, degrees=False).as_matrix()
        mtx[:3, 3] = tvec
        T_base2tool.append(mtx)

    # Project the object points over the "robot arm" into the image plane
    reprpoints = []
    for i in range(len(objpoints)): # Loop over each image
        objp = np.hstack((objpoints[i], np.ones((objpoints[i].shape[0], 1)))) # Create homogenous coordinates
        transformed_objp = T_tool2cam @ T_base2tool[i] @ np.linalg.inv(T_base2world) @ objp.T # Transform Object Points from WCS --> BCS --> TCS --> CCS
        transformed_objp = transformed_objp.T[:, :3] # Remove homogenous component
        reprojected_points, _ = cv2.projectPoints(transformed_objp, np.array([0., 0., 0.]), np.array([0., 0., 0.]), camera_matrix, distortion) # transformed_objpoints are already in CCS
        reprpoints.append(reprojected_points)

    # Calculate residuals
    imgpoints = np.vstack(imgpoints).reshape(-1)
    reprpoints = np.vstack(reprpoints).reshape(-1)
    residuals = imgpoints - reprpoints

    return residuals

def refine_hand_eye_pose(dataset_path):
    # Read Hand-Eye-Pose and Base-World-Pose from linear approach
    T_tool2cam = np.loadtxt(os.path.join(dataset_path, "sparse", "0", "T_tool2cam.txt"))
    T_base2world = np.loadtxt(os.path.join(dataset_path, "sparse", "0", "T_base2world.txt"))
    rvec_tool2cam = R.from_matrix(T_tool2cam[:3, :3]).as_euler("xyz", degrees=False)
    tvec_tool2cam = T_tool2cam[:3, 3]
    rvec_base2world = R.from_matrix(T_base2world[:3, :3]).as_euler("xyz", degrees=False)
    tvec_base2world = T_base2world[:3, 3]

    # Read MoCap poses and invert them
    R_tool2base, t_tool2base = read_poses(os.path.join(dataset_path, "sparse", "0", "images_mocap.txt"))
    R_base2tool = [np.linalg.inv(entry) for entry in R_tool2base]
    t_base2tool = [-entry_r @ entry_t for entry_r, entry_t in zip(R_base2tool, t_tool2base)]

    # Build parameter list (alpha, beta, gamma, tx, ty, tz for each pose)
    params_init = np.hstack([rvec_tool2cam, tvec_tool2cam, rvec_base2world, tvec_base2world])
    for Rmat, tvec in zip(R_base2tool, t_base2tool):
        rvec = R.from_matrix(Rmat).as_euler("xyz", degrees=False)
        params_init = np.hstack([params_init, rvec, tvec])

    # Read object points and image points
    objpoints, imgpoints = read_calib_points(dataset_path)

    # Read intrinsics
    fx, fy, cx, cy, k1, k2, p1, p2 = read_intrinsics(dataset_path)
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
    distortion = np.array([k1, k2, p1, p2])

    # Print reprojection error before optimization
    res = residuals(params_init, objpoints, imgpoints, camera_matrix, distortion)
    total_points = len(res) / 2
    reprojection_error = np.sqrt(np.sum(res ** 2) / total_points)
    print(f"Reprojection error before optimization: {reprojection_error:.4f}")

    # Perform non-linear optimization
    result = least_squares(residuals, 
                           params_init, 
                           args=(objpoints, imgpoints, camera_matrix, distortion),
                           jac="3-point",
                           method="lm",
                           loss="linear")

    # Print reprojection error after optimization
    params_opt = result["x"]
    res = residuals(params_opt, objpoints, imgpoints, camera_matrix, distortion)
    reprojection_error = np.sqrt(np.sum(res ** 2) / total_points)
    print(f"Reprojection error after optimization: {reprojection_error:.4f}")

    # Save optimized Hand-Eye-Pose and Base-World-Pose
    T_tool2cam_opt = np.eye(4)
    T_tool2cam_opt[:3, :3] = R.from_euler("xyz", params_opt[:3], degrees=False).as_matrix()
    T_tool2cam_opt[:3, 3] = params_opt[3:6]
    T_base2world_opt = np.eye(4)
    T_base2world_opt[:3, :3] = R.from_euler("xyz", params_opt[6:9], degrees=False).as_matrix()
    T_base2world_opt[:3, 3] = params_opt[9:12]

    np.savetxt(os.path.join(dataset_path, "sparse", "0", "T_tool2cam_opt.txt"), T_tool2cam_opt, fmt="%.6f")
    np.savetxt(os.path.join(dataset_path, "sparse", "0", "T_base2world_opt.txt"), T_base2world_opt, fmt="%.6f")
    print(f"[✓] Saved refined Hand-Eye-Pose to {os.path.join(dataset_path, "sparse", "0", "T_tool2cam_opt.txt")}")
    print(f"[✓] Saved refined Base-World-Pose to {os.path.join(dataset_path, "sparse", "0", "T_base2world_opt.txt")}")
