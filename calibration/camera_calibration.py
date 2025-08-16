import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as R
from calibration.utils import findChessboardCorners

def perform_camera_calibration(dataset_path):
    """
    This function performs camera calibration using images of a checkerboard pattern captured in the dataset.
    It saves the camera intrinsics and extrinsics to files.

    Args:
        dataset_path (str): The path where the dataset is saved, which contains the images of the checkerboard pattern.
    """
    # Define Chessboard
    chessboard = {'num_corners_down' : 23,
                    'num_corners_right' : 16,
                    'origin_marker_pos_down' : 10,
                    'origin_marker_pos_right' : 7,
                    'square_size' : 16}
    cd = chessboard['num_corners_down']
    cr = chessboard['num_corners_right']
    ss = chessboard['square_size']

    # Define Object Points in meters
    objp = np.zeros((cd*cr, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cr*ss:ss, 0:cd*ss:ss].T.reshape(-1, 2) 
    objp = objp / 1000

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d point in image plane

    image_folder = os.path.join(dataset_path, "images")

    for filename in sorted(os.listdir(image_folder)):
        filepath = os.path.join(image_folder, filename)
        print(filepath)
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chess board corners
        retval, found_shape, corners, ids = findChessboardCorners(gray, chessboard)
        if not retval:
            print(f"Checkerboard not found in {filename}. Please delete this image and run the script again.")

        # If found, add the object points and image points
        if retval:
            visible_objp = np.array([objp[i] for i in ids])
            objpoints.append(visible_objp)
            imgpoints.append(corners)

            # Draw the corners with their ids
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 255, 0)
            thickness = 2
            for idx, corner in enumerate(corners):
                x, y = corner.ravel().astype(int)
                text = str(ids[idx])
                cv2.putText(img, text, (x + 5, y - 5), font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.drawChessboardCorners(img, found_shape, corners, retval)
            scale = 0.3
            resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            cv2.imshow('img', resized_img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Calibrate flags for OPENCV camera model
    flags_opencv = (cv2.CALIB_FIX_K3) # Coefficient k3 is not changed during the optimization.

    # Calibrate flags for PINHOLE camera model 
    flags_pinhole = (cv2.CALIB_FIX_PRINCIPAL_POINT | # Principal point is not changed during the global optimization and stays at the center 
        cv2.CALIB_ZERO_TANGENT_DIST | # Tangential distortion_fo coefficients (p1, p2) are set to zeros
        cv2.CALIB_FIX_K1 | # Corresponding radial distortion_fo coefficient is not changed during the optimization and set to zero.
        cv2.CALIB_FIX_K2 |
        cv2.CALIB_FIX_K3) 
    
    # Calibrate with FULL_OPENCV model for accurate pose estimation
    w, h = gray.shape[::-1]
    num_images = len(imgpoints)
    repr_error_o, camera_matrix_o, distortion_o, rvecs_o, tvecs_o = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None, flags=flags_opencv)
    print(f"[✓] Camera calibration (OPENCV) completed with reprojection error: {repr_error_o:.4f} px.")

    # Calibrate with SIMPLE_PINHOLE model for retrieval onthefly_nvs-fitting calibration
    repr_error_p, camera_matrix_p, distortion_p, rvecs_p, tvecs_p = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None, flags=flags_pinhole)
    print(f"[✓] Camera calibration (PINHOLE) completed with reprojection error: {repr_error_p:.4f} px.")

    # Save poses (from OPENCV-calibration)
    pose_file = os.path.join(dataset_path, "sparse", "0", "images_checkerboard.txt")
    with open(pose_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_images}, mean observations per image: 0\n")
        f.write(f"# These poses have been computed using the OPENCV calibration, therefore CAMERA_ID is set to 2\n")
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

    with open(intr_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, w, h, PARAMS[]\n")
        f.write("# Number of cameras: 2\n")
        f.write("# PARAMS for PINHOLE are: w, h, fx, fy, cx, cy\n")
        f.write("# PARAMS for OPENCV are: w, h, fx, fy, cx, cy, k1, k2, p1, p2\n")
        f.write(f"1 PINHOLE {w} {h} {fx_p:.6f} {fy_p:.6f} {cx_p:.6f} {cy_p:.6f}\n")
        f.write(f"2 OPENCV {w} {h} {fx_o:.6f} {fy_o:.6f} {cx_o:.6f} {cy_o:.6f} {k1:.6f} {k2:.6f} {p1:.6f} {p2:.6f}\n")

    print(f"[✓] Saved intrinsics to {intr_file}")
    