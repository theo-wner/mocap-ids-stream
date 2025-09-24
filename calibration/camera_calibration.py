import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as R
from calibration.utils import findChessboardCorners

def perform_camera_calibration(dataset_path, chessboard):
    """
    This function performs camera calibration using images of a checkerboard pattern captured in the dataset.
    It saves the camera intrinsics and extrinsics to files.

    Args:
        dataset_path (str): The path where the dataset is saved, which contains the images of the checkerboard pattern.
    """
    image_folder = os.path.join(dataset_path, "images")
    objpoints, imgpoints, point_ids = findChessboardCorners(image_folder, chessboard)

    # Calibrate with OPENCV model for accurate pose estimation
    img = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0]))
    w, h = img.shape[1], img.shape[0]
    num_images = len(imgpoints)
    flags_opencv = (cv2.CALIB_FIX_K3) # Coefficient k3 is not changed during the optimization.
    repr_error_o, camera_matrix_o, distortion_o, rvecs_o, tvecs_o = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None, flags=flags_opencv)

    # Calculate custom reprojection error to be sure of the implementation
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        reprojected_points, _ = cv2.projectPoints(objpoints[i], rvecs_o[i], tvecs_o[i], camera_matrix_o, distortion_o)
        total_error += np.sum(np.abs(imgpoints[i] - reprojected_points)**2)
        total_points += len(objpoints[i])
    mean_error = np.sqrt(total_error / total_points)
    print(f"[✓] Camera calibration (OPENCV) completed with reprojection error: {mean_error:.2f} px.")

    # Calibrate with PINHOLE model for retrieval onthefly_nvs-fitting calibration
    flags_pinhole = (cv2.CALIB_FIX_PRINCIPAL_POINT | # Principal point is not changed during the global optimization and stays at the center 
        cv2.CALIB_ZERO_TANGENT_DIST | # Tangential distortion_fo coefficients (p1, p2) are set to zeros
        cv2.CALIB_FIX_K1 | # Corresponding radial distortion_fo coefficient is not changed during the optimization and set to zero.
        cv2.CALIB_FIX_K2 |
        cv2.CALIB_FIX_K3) 
    repr_error_p, camera_matrix_p, distortion_p, rvecs_p, tvecs_p = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None, flags=flags_pinhole)
    
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        reprojected_points, _ = cv2.projectPoints(objpoints[i], rvecs_p[i], tvecs_p[i], camera_matrix_p, distortion_p)
        total_error += np.sum(np.abs(imgpoints[i] - reprojected_points)**2)
        total_points += len(objpoints[i])
    mean_error = np.sqrt(total_error / total_points)
    print(f"[✓] Camera calibration (PINHOLE) completed with reprojection error: {mean_error:.2f} px.")

    # Save poses (from OPENCV-calibration)
    pose_file = os.path.join(dataset_path, "sparse", "0", "images_checkerboard.txt")
    with open(pose_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_images}, mean observations per image: 0\n")
        f.write(f"# These poses have been computed using the OPENCV calibration\n")
        for i, (rvec, tvec) in enumerate(zip(rvecs_o, tvecs_o)):
            # Convert rotation vector to quaternion
            rotmat, _ = cv2.Rodrigues(rvec)
            quat = R.from_matrix(rotmat).as_quat()  # [x, y, z, w]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            tx, ty, tz = tvec.ravel()
            
            # Get original image name
            image_name = sorted(os.listdir(image_folder))[i]
            image_idx = int(image_name.split('.')[0])

            f.write(f"{image_idx} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} 2 {image_name}\n\n")

    print(f"[✓] Saved poses to {pose_file}")

    # Save intrinsics (OPENCV as well as PINHOLE)
    intr_file = os.path.join(dataset_path, "sparse", "0", "cameras.txt")
    fx_o = camera_matrix_o[0, 0]
    fy_o = camera_matrix_o[1, 1]
    cx_o = camera_matrix_o[0, 2]
    cy_o = camera_matrix_o[1, 2]
    fx_p = camera_matrix_p[0, 0]
    fy_p = camera_matrix_p[1, 1]
    cx_p = camera_matrix_p[0, 2]
    cy_p = camera_matrix_p[1, 2]
    
    k1, k2, p1, p2 = distortion_o.ravel()[:4]  # Assumes only 4 distortion_o coefficients

    # Save Calibration results and Image points for the following Hand-Eye-Calibration
    with open(intr_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, w, h, PARAMS[]\n")
        f.write("# Number of cameras: 2\n")
        f.write("# PARAMS for PINHOLE are: w, h, fx, fy, cx, cy\n")
        f.write("# PARAMS for OPENCV are: w, h, fx, fy, cx, cy, k1, k2, p1, p2\n")
        f.write(f"1 PINHOLE {w} {h} {fx_p:.6f} {fy_p:.6f} {cx_p:.6f} {cy_p:.6f}\n")
        f.write(f"2 OPENCV {w} {h} {fx_o:.6f} {fy_o:.6f} {cx_o:.6f} {cy_o:.6f} {k1:.6f} {k2:.6f} {p1:.6f} {p2:.6f}\n")
    print(f"[✓] Saved intrinsics to {intr_file}")

    imgpoints_file = os.path.join(dataset_path, "sparse", "0", "imgpoints.txt")
    with open(imgpoints_file, "w") as f:
        f.write("# IMAGE_ID POINT_ID X (right) Y (down)\n")
        for i, (ids_per_image, points_per_image) in enumerate(zip(point_ids, imgpoints)):
            image_name = sorted(os.listdir(image_folder))[i]
            image_idx = int(image_name.split('.')[0])
            for id, point in zip(ids_per_image, points_per_image):
                f.write(str(image_idx) + " " + str(id) + " " + " ".join(f"{x:.6f}" for x in point[0]) + "\n")
    print(f"[✓] Saved extracted image points for each image to {imgpoints_file}")
