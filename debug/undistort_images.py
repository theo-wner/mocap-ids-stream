import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from calibration.utils import read_poses, read_intrinsics
from calibration.utils import findChessboardCorners

# Define paths
dataset_path = "./data/calibrations/calibration_2025-09-22_10-11-25"
images_path = os.path.join(dataset_path, "images")
intrinsics_path = os.path.join(dataset_path, "sparse", "0", "cameras.txt")
undistorted_path = os.path.join(dataset_path, "undistorted")

# Get OPENCV intrinsics
w, h, fx, fy, cx, cy, k1, k2, p1, p2 = read_intrinsics(dataset_path)
camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])
distortion = np.array([k1, k2, p1, p2])

# Set up new camera matrix to be sure the principal point is in the image center
fx = camera_matrix[0][0]
fy = camera_matrix[1][1]
cx = (w / 2)
cy = (h / 2)
new_camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])

# Undistort all images
if not os.path.exists(undistorted_path):
    os.makedirs(undistorted_path)
    for img_file in sorted(os.listdir(images_path)):
        img_path = os.path.join(images_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[warn] Could not read image: {img_path}")
            continue

        undistorted = cv2.undistort(img, camera_matrix, distortion, None, new_camera_matrix)

        cv2.imwrite(os.path.join(undistorted_path, img_file), undistorted)

# Define Chessboard
chessboard = {'num_corners_down' : 24,
                'num_corners_right' : 17,
                'origin_marker_pos_down' : 11,
                'origin_marker_pos_right' : 7,
                'square_size' : 30}

# Find Corners
objpoints, imgpoints, point_ids = findChessboardCorners(undistorted_path, chessboard)

# Get extrinsics
R_mtcs, tvecs = read_poses(os.path.join(dataset_path, "sparse", "0", "images_checkerboard.txt"))
rvecs = []
for R_mtx in R_mtcs:
    rvec, _ = cv2.Rodrigues(R_mtx)
    rvecs.append(rvec)

# Calculate reprojection error
no_distortion = np.array([0., 0., 0., 0.])
total_error = 0
total_points = 0
for i in range(len(objpoints)):
    reprojected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], new_camera_matrix, no_distortion)
    total_error += np.sum(np.abs(imgpoints[i] - reprojected_points)**2)
    total_points += len(objpoints[i])
mean_error = np.sqrt(total_error / total_points)
print(f"Projected object points into image space with no distortion and reprojection error: {mean_error:.2f} px.")


