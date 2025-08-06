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

    # Calibrate
    repr_error, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f"[✓] Camera calibration completed with reprojection error: {repr_error:.4f} px")

    # Save poses
    pose_file = os.path.join(dataset_path, 'checkerboard_poses.txt')
    with open(pose_file, 'w') as f:
        f.write("IMAGE_ID QW QX QY QZ TX TY TZ IMAGE_NAME\n")
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # Convert rotation vector to quaternion
            rotmat, _ = cv2.Rodrigues(rvec)
            quat = R.from_matrix(rotmat).as_quat()  # [x, y, z, w]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            tx, ty, tz = tvec.ravel()
            
            # Get original image name
            image_name = sorted(os.listdir(image_folder))[i]
            image_idx = int(image_name.split('.')[0])

            f.write(f"{image_idx} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {image_name}\n")

    print(f"[✓] Saved poses to {pose_file}")

    # Save intrinsics
    intr_file = os.path.join(dataset_path, 'intrinsics.txt')
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = distortion.ravel()[:5]  # Assumes only 5 distortion coefficients

    with open(intr_file, 'w') as f:
        f.write("FX FY CX CY K1 K2 P1 P2 K3\n")
        f.write(f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {k1:.6f} {k2:.6f} {p1:.6f} {p2:.6f} {k3:.6f}\n")

    print(f"[✓] Saved intrinsics to {intr_file}")